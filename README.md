# Live Transcriber

Live speech-to-text desktop app (Tkinter) with six selectable engines,
automatic speaker labels, a proper dark mode, window transparency and a
caption-style overlay mode.

## Engines

| Engine | Type | Notes |
|---|---|---|
| **Vosk** | Offline | True streaming — words appear live as you speak. Light on CPU/RAM. |
| **Sherpa (streaming)** | Offline | Modern streaming Zipformer models — live partials like Vosk with better accuracy. Small (~90MB) and full (~600MB) English models downloadable in Settings. |
| **Whisper (faster-whisper)** | Offline | Best accuracy. Newest free models: `large-v3`, `large-v3-turbo`, `distil-large-v3`, plus all classic sizes. Models download on first use. |
| **Moonshine** | Offline | New (2024/25) English model built specifically for live captioning — very fast on CPU. `tiny` or `base`. |
| **Google Web Speech** | Online | **Free, zero setup** — no key or account. Sends audio to Google. Language set in Settings → Online. |
| **Google Cloud** | Online | Needs a service-account credentials JSON. Sends audio to Google. |

Streaming engines (Vosk, Sherpa) show live partial words; the others
transcribe phrase-by-phrase (a phrase ends after the configurable pause).

## Speaker labels (voice recognition)

With **Settings → Speakers** enabled and the speaker model downloaded
(~28MB, one click), every transcript line is tagged **Speaker 1**,
**Speaker 2**, … in a distinct colour. It works with *every* engine: each
utterance's voice embedding (sherpa-onnx WeSpeaker model) is clustered
against the speakers heard so far.

Tuning: lower the similarity threshold if one person gets split into several
speakers; raise it if different people get merged. "Reset learned speakers"
starts the numbering fresh.

## Install

Python 3.9+ (tested on 3.14).

```
pip install -r requirements.txt
```

The app degrades gracefully — any engine whose library isn't installed just
shows as unavailable. GPU: for faster-whisper on NVIDIA, install CUDA/cuDNN
and set Device=`cuda`, Compute=`float16`.

## Run

```
python Transcriber_GUI.py
```

1. Pick an **audio source** (⟳ rescans). Three kinds are listed:
   - regular inputs (microphones, GoXLR mixes, …),
   - **🔊 System audio** — whatever that output is playing, captured via
     WASAPI loopback (each GoXLR channel appears separately: Chat is ideal
     for transcribing the other people in a meeting),
   - **🎙+🔊 Mic + System audio** — default mic and default output mixed,
     so a meeting transcript includes both sides with one click.
2. Optionally pick a second source in **+ Mix with:** to combine *any* two
   sources into one transcript — input + output (your mic + what you hear),
   two inputs (two mics), or two outputs (two GoXLR channels). The first
   source sets the rate; the second is resampled and summed in. Leave it on
   **(nothing)** for a single source.
3. Pick an **engine** and press **Start**.

Sources are re-resolved by name when you press Start, and a failed open
re-initializes audio and retries automatically — so device renumbering
(e.g. the GoXLR utility restarting) no longer causes WDM-KS errors.

Toolbar extras: theme toggle (dark/light), 📌 always-on-top, opacity slider,
**Overlay mode** (hides controls, pins on top, semi-transparent — **Esc**
restores), Clear, Save transcript.

## Capturing system audio

Built in: the 🔊 System audio entries record any output device directly
(WASAPI loopback via PyAudioWPatch), and 🎙+🔊 mixes the default mic on top —
no Stereo Mix or virtual cables needed. Works the same on machines with a
single mic + speakers as it does with a GoXLR.

## Configuration

Everything lives in `config.ini` next to the script and is editable from
Settings: engine, Vosk/Sherpa model downloads + custom folders, Whisper and
Moonshine options, Google language/credentials, speaker-label settings,
phrase segmentation (pause threshold, max phrase length, energy threshold),
logging, font size and overlay opacity.

## Adding future models (no code changes)

The app is built to absorb new models three ways:

1. **`models.json`** — copy `models.example.json` to `models.json` next to the
   app/exe and add entries. New Vosk or Sherpa models appear in Settings with
   their own download button; extra faster-whisper names appear in the model
   dropdown. Sources: [Vosk models](https://alphacephei.com/vosk/models),
   [Sherpa models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models).
2. **Whisper model box is free-text** — with the faster-whisper backend you can
   type *any* Hugging Face CTranslate2 repo id (e.g.
   `deepdml/faster-whisper-large-v3-turbo-ct2`). When a new Whisper-family
   model is released, it works the day someone publishes a CT2 conversion.
3. **Engine registry (for code)** — every engine is one entry in the `ENGINES`
   dict in `Transcriber_GUI.py` plus a `prepare_*()` function. The dropdown,
   Settings page and validation generate themselves from the registry, so a
   brand-new engine type is a ~30-line addition in one place.

## Building & shipping to friends

```
powershell -ExecutionPolicy Bypass -File build.ps1
```

Run it from the repo root. It runs PyInstaller (`LiveTranscriber.spec`) and
writes `release\LiveTranscriber\` (the unzipped app) plus
`release\LiveTranscriber-win64.zip` (~165 MB). Friends just **unzip anywhere
writable** (not Program Files) and run `LiveTranscriber.exe` — no Python
needed. Models download next to the exe on first use; to save them the wait,
you can copy your `vosk_models/`, `sherpa_models/` and `speaker_models/`
folders into their unzipped folder.

`release/` is gitignored — it is regenerable, and the zip exceeds GitHub's
100 MB file limit. To distribute a build, attach the zip to a
[GitHub Release](https://github.com/Dorkling/Live-Transcript/releases)
rather than committing it.

Notes for recipients:
- Windows SmartScreen will warn about an unsigned exe — "More info → Run
  anyway". (Code-signing certificates fix this but cost money.)
- If an engine is missing on their machine, check `app_errors.log` next to
  the exe — the first line lists which engines loaded.

## Repo layout

Tracked in git (everything needed to build from scratch):

- `Transcriber_GUI.py` — the app, a single file.
- `LiveTranscriber.spec`, `build.ps1` — packaging.
- `requirements.txt` — dependencies.
- `models.example.json` — template for adding models without code changes.

Generated locally and gitignored:

- `vosk_models/`, `sherpa_models/`, `speaker_models/` — downloaded on demand
  from Settings; hundreds of MB, never committed.
- `release/`, `build/` — build output.
- `config.ini` — your settings; recreated with defaults on first run.
- `*.log` — transcript log and `app_errors.log`.

## License

MIT — see [LICENSE](LICENSE).

The speech engines and their models are separate projects under their own
(permissive) licenses — Vosk and sherpa-onnx are Apache 2.0, faster-whisper
and Moonshine are MIT, SpeechRecognition is BSD. A packaged build bundles
those libraries, so their notices apply to the distributed exe as well.
