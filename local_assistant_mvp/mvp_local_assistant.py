#!/usr/bin/env python3
import os, json, time, subprocess, wave, pathlib, atexit, shutil, random, shlex, signal
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import sounddevice as sd
import webrtcvad
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

# ------------ Config ------------
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_AUTOSTART = os.getenv("OLLAMA_AUTOSTART", "true").strip().lower() in {"1", "true", "yes", "on"}
OLLAMA_START_CMD = os.getenv("OLLAMA_START_CMD", "").strip()
OLLAMA_START_TIMEOUT = float(os.getenv("OLLAMA_START_TIMEOUT", "30"))
OLLAMA_RETRY_MAX = int(os.getenv("OLLAMA_RETRY_MAX", "10"))
OLLAMA_AUTO_PULL = os.getenv("OLLAMA_AUTO_PULL", "false").strip().lower() in {"1", "true", "yes", "on"}
ASR_MODEL    = os.getenv("ASR_MODEL", "small-int8")

PIPER_PATH   = os.path.expanduser(os.getenv("PIPER_PATH", ""))
PIPER_VOICE  = os.path.expanduser(os.getenv("PIPER_VOICE", ""))

WORKSPACE    = pathlib.Path(os.path.expanduser(os.getenv("WORKSPACE_DIR","~/ai_workspace"))).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

VAD_AGGR     = int(os.getenv("VAD_AGGR","2"))
SAMPLE_RATE  = int(os.getenv("SAMPLE_RATE","16000"))

# ------------ Utility ------------
def safe_path(p: str) -> pathlib.Path:
    # Resolve and ensure within WORKSPACE
    target = (WORKSPACE / p).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        raise PermissionError("Path outside sandbox.")
    return target

def list_dir(path="."):
    target = safe_path(path)
    items = []
    for entry in sorted(target.iterdir()):
        items.append({
            "name": entry.name,
            "is_dir": entry.is_dir(),
            "size": entry.stat().st_size if entry.is_file() else None
        })
    return {"cwd": str(target), "items": items}

def read_file(path):
    target = safe_path(path)
    if not target.is_file():
        return {"error": "not a file"}
    # Cap file size
    if target.stat().st_size > 512*1024:
        return {"error": "file too large (>512KB)"}
    return {"path": str(target), "content": target.read_text(encoding="utf-8", errors="replace")}

def write_file(path, content):
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"written": str(target), "bytes": len(content.encode('utf-8'))}

def speak(text: str):
    if not (PIPER_PATH and PIPER_VOICE and os.path.exists(os.path.expanduser(PIPER_PATH))):
        return
    # create wav via piper, then play with sounddevice
    tmp_wav = WORKSPACE / "_tts_out.wav"
    try:
        proc = subprocess.run(
            [os.path.expanduser(PIPER_PATH), "-m", os.path.expanduser(PIPER_VOICE), "-f", str(tmp_wav)],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            return
        # play wav
        with wave.open(str(tmp_wav), 'rb') as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)
        sd.play(data, sr)
        sd.wait()
    finally:
        if tmp_wav.exists():
            try: tmp_wav.unlink()
            except: pass

# ------------ Audio/VAD ------------
class Recorder:
    def __init__(self, samplerate=SAMPLE_RATE, vad_aggr=VAD_AGGR):
        self.samplerate = samplerate
        self.vad = webrtcvad.Vad(vad_aggr)
        self.block_size = 160  # 10ms at 16kHz
        self.block_bytes = self.block_size * 2  # int16
        self.buffer = bytearray()
        self.speaking = False
        self.silence_ms = 0
        self.max_ms = 15000  # hard cap per utterance
        self.min_ms = 400    # minimum speech length

    def _callback(self, indata, frames, time_info, status):
        pcm16 = indata.tobytes()
        # split into 10ms frames for VAD
        for i in range(0, len(pcm16), self.block_bytes):
            frame = pcm16[i:i+self.block_bytes]
            if len(frame) < self.block_bytes:
                continue
            try:
                is_speech = self.vad.is_speech(frame, self.samplerate)
            except:
                is_speech = False
            if is_speech:
                self.buffer.extend(frame)
                self.speaking = True
                self.silence_ms = 0
            else:
                if self.speaking:
                    self.buffer.extend(frame)
                self.silence_ms += 10

    def record_utterance(self) -> bytes:
        self.buffer = bytearray()
        self.speaking = False
        self.silence_ms = 0

        with sd.RawInputStream(samplerate=self.samplerate, channels=1, dtype='int16', callback=self._callback, blocksize=self.block_size):
            t0 = time.time()
            print("🎙️ Sprich jetzt... (automatisches Ende nach kurzer Stille)")
            while True:
                elapsed_ms = int((time.time() - t0) * 1000)
                if self.speaking and self.silence_ms > 600 and elapsed_ms > self.min_ms:
                    break
                if elapsed_ms > self.max_ms:
                    break
                time.sleep(0.05)
        return bytes(self.buffer)

def save_wav(pcm16: bytes, samplerate=SAMPLE_RATE, path: pathlib.Path=None) -> pathlib.Path:
    if path is None:
        path = WORKSPACE / "_last_in.wav"
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16)
    return path

# ------------ ASR ------------
class ASR:
    def __init__(self, model_size=ASR_MODEL, device="auto"):
        print(f"⏳ Lade ASR-Modell: {model_size}")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe(self, wav_path: pathlib.Path, language=None) -> str:
        segments, info = self.model.transcribe(str(wav_path), language=language, vad_filter=True)
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        return text

# ------------ LLM chat ------------
SYSTEM_PROMPT = '''Du bist ein lokaler Assistent, läufst offline bei der Nutzerin/dem Nutzer.
Wenn du eine Dateisystem-Operation brauchst, gib *genau* ein JSON-Objekt zurück, NUR dieses, ohne Text drumherum.
Formate:
{"tool":"list_dir","args":{"path":"."}}
{"tool":"read_file","args":{"path":"notes/todo.txt"}}
{"tool":"write_file","args":{"path":"notes/todo.txt","content":"Text"}}
Schreibe sonst normale Antworten auf Deutsch und sei knapp & hilfreich.
'''

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.autostart = OLLAMA_AUTOSTART
        self.start_timeout = max(1.0, OLLAMA_START_TIMEOUT)
        self.retry_max = max(1, OLLAMA_RETRY_MAX)
        self.auto_pull = OLLAMA_AUTO_PULL
        self._server_proc: Optional[subprocess.Popen] = None
        self._started_by_client = False
        self._model_ready = False
        self._orig_signal_handlers: Dict[int, Any] = {}
        if self.autostart:
            self.start_cmd, self._managed_start = self._resolve_start_cmd(OLLAMA_START_CMD)
        else:
            self.start_cmd, self._managed_start = [], False
        atexit.register(self._cleanup)
        self._register_signal_handlers()
        self._ensure_server_and_model()

    def _resolve_start_cmd(self, cmd_str: str) -> Tuple[List[str], bool]:
        if cmd_str:
            parts = shlex.split(cmd_str)
        else:
            ollama_bin = shutil.which("ollama")
            if not ollama_bin:
                raise RuntimeError("'ollama' nicht gefunden. Bitte Ollama installieren und in den PATH aufnehmen.")
            parts = [ollama_bin, "serve"]
        if parts[0] == "ollama":
            ollama_bin = shutil.which("ollama")
            if not ollama_bin:
                raise RuntimeError("'ollama' nicht gefunden. Bitte Ollama installieren und in den PATH aufnehmen.")
            parts[0] = ollama_bin
        managed = len(parts) >= 2 and os.path.basename(parts[0]) == "ollama" and parts[1] == "serve"
        return parts, managed

    def _register_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
                self._orig_signal_handlers[sig] = previous

                def _handler(signum, frame, *, previous_handler=previous):
                    self._cleanup()
                    if callable(previous_handler):
                        previous_handler(signum, frame)
                    elif previous_handler == signal.SIG_DFL:
                        signal.signal(signum, signal.SIG_DFL)
                        os.kill(os.getpid(), signum)

                signal.signal(sig, _handler)
            except (ValueError, AttributeError):
                continue

    def _cleanup(self):
        if self._started_by_client and self._server_proc and self._server_proc.poll() is None:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=5)
            except Exception:
                try:
                    self._server_proc.kill()
                except Exception:
                    # Ignore exceptions during forced process kill; cleanup should not fail if process is already dead or cannot be killed.
                    pass

    def _fetch_tags(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=3)
        r.raise_for_status()
        return r.json()

    def _wait_for_server(self) -> Optional[Dict[str, Any]]:
        deadline = time.time() + self.start_timeout
        attempt = 0
        delay = 0.5
        while attempt < self.retry_max and time.time() < deadline:
            attempt += 1
            try:
                return self._fetch_tags()
            except requests.RequestException:
                if time.time() >= deadline:
                    break
                jitter = random.uniform(0.05, 0.3)
                sleep_for = min(delay + jitter, max(0.0, deadline - time.time()))
                time.sleep(max(0.05, sleep_for))
                delay = min(delay * 1.5, 3.0)
        return None

    def _ensure_server_and_model(self) -> None:
        tags = self._wait_for_server()
        if tags is None:
            if not self.autostart:
                raise RuntimeError("Ollama-Server nicht erreichbar (Server down). Bitte manuell starten oder OLLAMA_AUTOSTART aktivieren.")
            self._start_server()
            tags = self._wait_for_server()
            if tags is None:
                raise RuntimeError("Ollama-Server konnte nicht gestartet werden. Bitte Installation prüfen oder manuell mit 'ollama serve' starten.")
        self._ensure_model_available(tags)

    def _start_server(self) -> None:
        if self._managed_start:
            if self._server_proc and self._server_proc.poll() is None:
                return
            try:
                self._server_proc = subprocess.Popen(
                    self.start_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._started_by_client = True
            except Exception as exc:
                raise RuntimeError(f"Konnte Ollama-Server nicht starten: {exc}") from exc
        else:
            try:
                subprocess.run(self.start_cmd, check=True)
            except Exception as exc:
                raise RuntimeError(f"Startkommando für Ollama fehlgeschlagen: {exc}") from exc

    def _model_matches(self, entry: Dict[str, Any]) -> bool:
        name = entry.get("name")
        if not name:
            return False
        if self.model == name:
            return True
        tags = entry.get("tags") or []
        for tag in tags:
            if self.model == f"{name}:{tag}":
                return True
        return False

    def _ensure_model_available(self, tags: Optional[Dict[str, Any]] = None) -> None:
        if self._model_ready:
            return
        tags = tags or self._fetch_tags()
        models = tags.get("models", []) if isinstance(tags, dict) else []
        if any(self._model_matches(m) for m in models):
            self._model_ready = True
            return
        if not self.auto_pull:
            raise RuntimeError(f"Modell '{self.model}' ist nicht verfügbar. Bitte mit 'ollama pull {self.model}' installieren oder OLLAMA_AUTO_PULL aktivieren.")
        print(f"⬇️ Ziehe fehlendes Ollama-Modell '{self.model}' ...")
        self._pull_model()
        tags = self._wait_for_server()
        if tags is None or not any(self._model_matches(m) for m in tags.get("models", [])):
            raise RuntimeError(f"Modell '{self.model}' konnte nicht bereitgestellt werden. Bitte Installation prüfen.")
        self._model_ready = True

    def _pull_model(self) -> None:
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            raise RuntimeError("'ollama' nicht gefunden. Automatischer Pull nicht möglich.")
        try:
            subprocess.run([ollama_bin, "pull", self.model], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"'ollama pull {self.model}' fehlgeschlagen: {exc}") from exc

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if not self._model_ready:
            self._ensure_model_available()
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        except requests.RequestException as exc:
            raise RuntimeError("Fehler: Ollama-Server nicht erreichbar (Server down?).") from exc
        if r.status_code == 404:
            raise RuntimeError(f"Fehler: Modell '{self.model}' nicht gefunden. Bitte installieren oder OLLAMA_AUTO_PULL aktivieren.")
        if r.status_code >= 500:
            raise RuntimeError(f"Fehler: Ollama-Server meldet einen internen Fehler ({r.status_code}).")
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError as exc:
            raise RuntimeError("Fehler: Ungültige Antwort vom Ollama-Server erhalten.") from exc
        if "error" in data:
            error_msg = str(data["error"]).strip()
            lowered = error_msg.lower()
            if "not found" in lowered:
                raise RuntimeError(f"Fehler: Modell '{self.model}' nicht gefunden. Bitte installieren oder OLLAMA_AUTO_PULL aktivieren.")
            if "loading" in lowered:
                raise RuntimeError(f"Fehler: Modell '{self.model}' lädt noch. Bitte etwas warten und erneut versuchen.")
            raise RuntimeError(f"Fehler vom Ollama-Server: {error_msg}")
        return data.get("message", {}).get("content", "")

def maybe_parse_tool(s: str) -> Optional[Dict[str,Any]]:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                return obj
        except json.JSONDecodeError:
            return None
    return None

def main():
    print("=== Local Assistant MVP ===")
    print(f"Workspace: {WORKSPACE}")
    rec = Recorder()
    asr = ASR()
    ollama = OllamaClient()

    messages = [{"role":"system","content": SYSTEM_PROMPT}]

    while True:
        input("↩︎ ENTER drücken und sprechen… (Ctrl+C zum Beenden) ")
        pcm = rec.record_utterance()
        if len(pcm) < 3200:
            print("…nichts gehört. Nochmal!")
            continue
        wavp = save_wav(pcm)
        text = asr.transcribe(wavp, language=None)
        print(f"🗣️ Du: {text}")
        messages.append({"role":"user","content": text})

        reply = ollama.chat(messages)
        tool_req = maybe_parse_tool(reply)

        if tool_req:
            tool = tool_req.get("tool")
            args = tool_req.get("args",{})
            try:
                if tool == "list_dir":
                    res = list_dir(args.get("path","."))
                elif tool == "read_file":
                    res = read_file(args.get("path",""))
                elif tool == "write_file":
                    res = write_file(args.get("path",""), args.get("content",""))
                else:
                    res = {"error": f"unknown tool {tool}"}
            except Exception as e:
                res = {"error": str(e)}
            # feed result back to model
            messages.append({"role":"user","content": f"TOOL_RESULT:\n{json.dumps(res, ensure_ascii=False, indent=2)}"})
            reply = ollama.chat(messages)

        print(f"🤖 Assistent: {reply}")
        speak(reply)
        messages.append({"role":"assistant","content": reply})

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAuf Wiedersehen!")