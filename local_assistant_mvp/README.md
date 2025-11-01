# Local Assistant MVP (Linux)

Ein minimal lauffähiges MVP: Mikrofon → ASR (faster-whisper) → LLM (Ollama) → TTS (Piper) → Lautsprecher.
Plus: drei abgesicherte Datei-Tools (`list_dir`, `read_file`, `write_file`) in einem Sandbox-Ordner.

## Quickstart
```bash
# 1) Systemvoraussetzungen (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y python3-venv python3-dev portaudio19-dev

# 2) Piper (Binary) + deutsche Stimme (Beispiel)
# Lade piper von https://github.com/rhasspy/piper/releases und entpacke nach ~/bin
# Lade eine de-DE Stimme (z.B. "de_DE-thorsten-high.onnx.gz") nach ~/voices
# Beispiel-Pfade: PIPER_PATH=~/bin/piper  PIPER_VOICE=~/voices/de_DE-thorsten-high.onnx.gz

# 3) Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# 4) Repo & Python-Umgebung
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5) Umgebung konfigurieren
cp .env.example .env
# prüfe/ändere Variablen in .env (PIPER_PATH, PIPER_VOICE, WORKSPACE_DIR usw.)

# 6) Start
python mvp_local_assistant.py
```
Hinweis: Das Skript funktioniert ohne TTS (spricht dann nicht), falls `PIPER_PATH`/`PIPER_VOICE` fehlen.

## Bedienung
- ENTER: Eine Äußerung aufnehmen. Ende automatisch nach kurzer Stille.
- Das Transkript geht ans LLM. Antworten erscheinen im Terminal und werden gesprochen (falls TTS konfiguriert).
- Tools: Der Assistent kann **explizit** Tools anfordern, indem er *genau* dieses JSON zurückgibt:
  ```json
  {"tool":"list_dir","args":{"path":"."}}
  ```
  Erlaubt sind: `list_dir`, `read_file`, `write_file` (nur im Sandbox-Verzeichnis).

## Bekannte Grenzen
- Echtzeit (WebRTC) ist hier bewusst nicht enthalten; erst stabilisieren, dann LiveKit integrieren.
- VAD ist simpel (webrtcvad); bei sehr lauten Umgebungen ggf. `VAD_AGGR` in `.env` anpassen.