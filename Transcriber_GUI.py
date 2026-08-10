#!/usr/bin/env python3
"""Live Transcriber.

Live speech-to-text with selectable engines:
  * Vosk          - offline, true streaming with live partial results
  * Whisper       - offline, via faster-whisper (recommended) or openai-whisper
  * Google Cloud  - online, requires a credentials JSON file

Extras: proper dark/light theming, window transparency, always-on-top pin and
a caption-style overlay mode.

All audio capture goes through sounddevice with a single recorder thread, so
every engine sees the same device list and the same indices.
"""

import sys
import os
import json
import queue
import threading
import time
import zipfile
import tarfile
import glob
import importlib.util
import configparser
from collections import deque
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font as tkfont

# In a windowed PyInstaller build there is no console; stray prints from
# libraries would raise. Send them to a log file next to the exe instead.
if getattr(sys, "frozen", False):
    try:
        _frozen_log = open(
            os.path.join(os.path.dirname(sys.executable), "app_errors.log"),
            "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = _frozen_log
        sys.stderr = _frozen_log
    except Exception:
        pass


def _fatal_startup_error(message):
    """Show a startup error even if the main GUI never got built."""
    print(f"[FATAL] {message}")
    try:
        hidden = tk.Tk()
        hidden.withdraw()
        messagebox.showerror("Live Transcriber - startup error", message)
        hidden.destroy()
    except Exception:
        pass
    sys.exit(1)


# --- Required dependencies -------------------------------------------------
try:
    import numpy as np
except ImportError:
    _fatal_startup_error("The 'numpy' package is required.\n\nInstall it with:\n    pip install numpy")

try:
    import sounddevice as sd
except Exception as e:
    _fatal_startup_error(f"The 'sounddevice' package (and PortAudio) is required.\n\nInstall it with:\n    pip install sounddevice\n\nDetails: {e}")

# --- Optional dependencies (engines degrade gracefully) ---------------------
try:
    import vosk
    HAVE_VOSK = True
except Exception:
    vosk = None
    HAVE_VOSK = False

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    HAVE_FASTER_WHISPER = True
except Exception:
    FasterWhisperModel = None
    HAVE_FASTER_WHISPER = False

try:
    import whisper as openai_whisper
    HAVE_OPENAI_WHISPER = True
except Exception:
    openai_whisper = None
    HAVE_OPENAI_WHISPER = False

try:
    import speech_recognition as sr_lib
    HAVE_SR = True
except Exception:
    sr_lib = None
    HAVE_SR = False

try:
    import sherpa_onnx
    HAVE_SHERPA = True
except Exception:
    sherpa_onnx = None
    HAVE_SHERPA = False

# WASAPI loopback (capture what the PC is playing) - Windows only.
try:
    import pyaudiowpatch as pyaudio_patch
    HAVE_LOOPBACK = sys.platform == "win32"
except Exception:
    pyaudio_patch = None
    HAVE_LOOPBACK = False

# Moonshine pulls in heavy imports; detect it cheaply and import on first use.
HAVE_MOONSHINE = importlib.util.find_spec("moonshine_onnx") is not None

try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    requests = None
    HAVE_REQUESTS = False


# --- Constants ---------------------------------------------------------------
APP_TITLE = "Live Transcriber"
CONFIG_FILE = "config.ini"
MODEL_BASE_DIR = "vosk_models"

MODEL_INFO = {
    "small": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "extracted_dir_name": "vosk-model-small-en-us-0.15",
        "description": "Small US English (~45MB) - fast, lower accuracy",
    },
    "large": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "extracted_dir_name": "vosk-model-en-us-0.22",
        "description": "Large US English (~1.8GB) - slower, better accuracy",
    },
    "gigaspeech": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip",
        "extracted_dir_name": "vosk-model-en-us-0.42-gigaspeech",
        "description": "Gigaspeech US English (~2.6GB) - slowest, best accuracy",
    },
}
DEFAULT_VOSK_MODEL_TYPE = "small"

SHERPA_BASE_DIR = "sherpa_models"
SHERPA_MODELS = {
    "small-en": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
        "dir_name": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "description": "Streaming Zipformer EN small (~90MB) - fast, light on CPU",
    },
    "full-en": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "dir_name": "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "description": "Streaming Zipformer EN full (~600MB) - best streaming accuracy",
    },
}
DEFAULT_SHERPA_MODEL = "small-en"

MOONSHINE_MODELS = ["moonshine/tiny", "moonshine/base"]
DEFAULT_MOONSHINE_MODEL = "moonshine/base"

SPEAKER_MODEL_DIR = "speaker_models"
SPEAKER_MODEL_FILE = "wespeaker_en_voxceleb_CAM++.onnx"
SPEAKER_MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                     "speaker-recongition-models/" + SPEAKER_MODEL_FILE)

FASTER_WHISPER_SIZES = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "distil-small.en", "medium", "medium.en", "distil-medium.en",
    "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3",
]
OPENAI_WHISPER_SIZES = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v3", "turbo",
]
DEFAULT_WHISPER_SIZE = "base"
WHISPER_DEVICES = ["auto", "cpu", "cuda"]
WHISPER_COMPUTE_TYPES = ["auto", "int8", "int8_float16", "float16", "float32"]

# Engine metadata lives in the ENGINES registry further down; ENGINE_LABELS
# is derived from it.

# --- Theme palettes ----------------------------------------------------------
LIGHT = {
    "bg": "#f3f4f6", "surface": "#ffffff", "control": "#e3e5ea",
    "control_hover": "#d4d7dd", "fg": "#1b1d23", "muted": "#6b7280",
    "border": "#c6c9d0", "accent": "#2563eb", "accent_fg": "#ffffff",
    "accent_hover": "#1d4ed8", "text_bg": "#ffffff", "text_fg": "#1b1d23",
    "danger": "#c62828", "sel_bg": "#cfe1ff",
    "speakers": ["#2563eb", "#1f8a4d", "#b97c14", "#a23bb0", "#0f8c8c", "#cc5a2a"],
}
DARK = {
    "bg": "#1e1f24", "surface": "#26272e", "control": "#33353d",
    "control_hover": "#3f424c", "fg": "#e7e8ea", "muted": "#9b9da6",
    "border": "#3c3e47", "accent": "#4f8ef7", "accent_fg": "#ffffff",
    "accent_hover": "#6da2f8", "text_bg": "#16171b", "text_fg": "#e7e8ea",
    "danger": "#ef6363", "sel_bg": "#314a73",
    "speakers": ["#4f8ef7", "#37b26c", "#e0a23c", "#d779e0", "#4fc3c3", "#ef8b63"],
}

# --- Globals -----------------------------------------------------------------
config = configparser.ConfigParser()
gui_queue = queue.Queue()
_model_cache = {}

DEFAULTS = {
    "Engine": {"type": "vosk", "google_cloud_credentials_json": ""},
    "Models": {"preferred_vosk_model_type": DEFAULT_VOSK_MODEL_TYPE,
               "model_directory": MODEL_BASE_DIR},
    "Paths": {"custom_model_path": ""},
    "Whisper": {"backend": "faster-whisper", "model_size": DEFAULT_WHISPER_SIZE,
                "device": "auto", "compute_type": "auto",
                "language": "auto", "vad_filter": "True"},
    "Sherpa": {"model": DEFAULT_SHERPA_MODEL, "custom_model_dir": ""},
    "Moonshine": {"model": DEFAULT_MOONSHINE_MODEL},
    "GoogleWeb": {"language": "en-US"},
    "Speakers": {"enabled": "True", "similarity_threshold": "0.45",
                 "max_speakers": "8"},
    "Audio": {"audio_source_name": "", "mix_source_name": "",
              "log_file": "live_transcription.log",
              "pause_threshold": "0.7", "max_phrase_sec": "12.0",
              "energy_threshold": "auto"},
    "Settings": {"enable_logging": "True", "theme": "dark",
                 "opacity": "100", "overlay_opacity": "85",
                 "always_on_top": "False", "font_size": "11"},
}


def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def load_config():
    path = os.path.join(get_base_path(), CONFIG_FILE)
    config.read_dict(DEFAULTS)
    if os.path.exists(path):
        try:
            config.read(path, encoding="utf-8")
        except Exception as e:
            print(f"Could not parse {path}: {e} - using defaults.")
    # Merge in any options added since the file was written.
    for section, options in DEFAULTS.items():
        if not config.has_section(section):
            config.add_section(section)
        for option, value in options.items():
            if not config.has_option(section, option):
                config.set(section, option, value)
    # Migrate the pre-rename audio key from very old configs.
    if config.has_option("Audio", "input_device_name") and not config.get("Audio", "audio_source_name", fallback=""):
        config.set("Audio", "audio_source_name", config.get("Audio", "input_device_name"))
        config.remove_option("Audio", "input_device_name")
    save_config()
    load_model_catalog()


def load_model_catalog():
    """Merge a models.json file (next to the app) into the built-in catalogs.

    Lets new models be added without touching code. Recognised keys:
      "vosk":   {key: {"url", "extracted_dir_name", "description"}, ...}
      "sherpa": {key: {"url", "dir_name", "description"}, ...}
      "faster_whisper_models": ["name-or-hf-repo-id", ...]
      "moonshine_models": ["moonshine/...", ...]
    See models.example.json for a template.
    """
    path = os.path.join(get_base_path(), "models.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, info in (data.get("vosk") or {}).items():
            if {"url", "extracted_dir_name", "description"} <= set(info):
                MODEL_INFO[key] = info
        for key, info in (data.get("sherpa") or {}).items():
            if {"url", "dir_name", "description"} <= set(info):
                SHERPA_MODELS[key] = info
        for name in data.get("faster_whisper_models") or []:
            if name not in FASTER_WHISPER_SIZES:
                FASTER_WHISPER_SIZES.append(name)
        for name in data.get("moonshine_models") or []:
            if name not in MOONSHINE_MODELS:
                MOONSHINE_MODELS.append(name)
        print(f"Loaded extra models from {path}")
    except Exception as e:
        print(f"Could not load models.json: {e}")


def save_config():
    path = os.path.join(get_base_path(), CONFIG_FILE)
    try:
        with open(path, "w", encoding="utf-8") as f:
            config.write(f)
        return True
    except OSError as e:
        print(f"Error saving config to {path}: {e}")
        return False


# --- Audio helpers -----------------------------------------------------------
MIX_SOURCE_LABEL = "\U0001F399+\U0001F50A Mic + System audio (default devices)"
NO_MIX_LABEL = "(nothing)"


def refresh_portaudio():
    """Re-initialize PortAudio so devices added/removed since startup appear.

    Device indices are assigned when PortAudio initializes; GoXLR-style
    virtual devices renumber everything when their utility restarts.
    """
    try:
        sd._terminate()
        sd._initialize()
    except Exception as e:
        print(f"PortAudio re-init failed: {e}")


def list_audio_devices():
    """Map of display name -> source descriptor.

    Descriptors are (kind, index, device_name):
      ("sd", i, name)        - regular input via sounddevice
      ("loopback", i, name)  - system audio via WASAPI loopback (pyaudiowpatch)
      ("mix", None, None)    - default mic + default system audio, mixed
    Indices are re-resolved by name when a session starts, so a stale list
    can't open the wrong device.
    """
    devices = {}
    try:
        all_devs = sd.query_devices()
        apis = sd.query_hostapis()
        # The default-input index usually points at an MME/DirectSound entry;
        # remember its *name* so the matching WASAPI entry gets marked too.
        default_name = None
        try:
            default_in = sd.default.device[0]
            if 0 <= default_in < len(all_devs):
                default_name = all_devs[default_in]["name"]
        except Exception:
            default_in = -1
        entries = []
        for i, d in enumerate(all_devs):
            if d.get("max_input_channels", 0) <= 0:
                continue
            api_name = apis[d["hostapi"]]["name"]
            entries.append((i, d, api_name))
        if sys.platform == "win32":
            wasapi = [e for e in entries if "WASAPI" in e[2]]
            if wasapi:
                entries = wasapi
        for i, d, api_name in entries:
            is_default = i == default_in or (
                default_name is not None
                and (d["name"] in default_name or default_name in d["name"]))
            mark = "  (default)" if is_default else ""
            devices[f"{i}: {d['name']} [{api_name}]{mark}"] = ("sd", i, d["name"])
    except Exception as e:
        print(f"Error listing audio devices: {e}")

    # System-audio (loopback) sources - what the PC is playing. Covers both
    # plain speaker/headphone setups and each GoXLR output channel.
    if HAVE_LOOPBACK:
        try:
            pa = pyaudio_patch.PyAudio()
            try:
                for d in pa.get_loopback_device_info_generator():
                    clean = d["name"].replace(" [Loopback]", "")
                    devices[f"\U0001F50A System audio: {clean}"] = ("loopback", int(d["index"]), d["name"])
            finally:
                pa.terminate()
        except Exception as e:
            print(f"Error listing loopback devices: {e}")
        if devices:
            devices[MIX_SOURCE_LABEL] = ("mix", None, None)
    return devices


def float_to_int16_bytes(audio_f32):
    return (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def resample_audio(audio_f32, rate_from, rate_to):
    if rate_from == rate_to or not len(audio_f32):
        return audio_f32.astype(np.float32, copy=False)
    n_out = max(1, int(round(len(audio_f32) * float(rate_to) / rate_from)))
    x_old = np.arange(len(audio_f32), dtype=np.float64)
    x_new = np.linspace(0.0, len(audio_f32) - 1, n_out)
    return np.interp(x_new, x_old, audio_f32).astype(np.float32)


def resample_to_16k(audio_f32, rate):
    return resample_audio(audio_f32, rate, 16000)


def get_energy_threshold():
    """Configured RMS threshold (0..1), or None to auto-calibrate."""
    raw = config.get("Audio", "energy_threshold", fallback="auto").strip().lower()
    if raw in ("", "auto"):
        return None
    try:
        return max(0.0005, float(raw))
    except ValueError:
        return None


def available_engines():
    return [key for key, spec in ENGINES.items() if spec["available"]()]


def resolve_vosk_model_path():
    """Return (path, error). One of the two is always None."""
    preferred = config.get("Models", "preferred_vosk_model_type", fallback=DEFAULT_VOSK_MODEL_TYPE)
    base = get_base_path()
    if preferred == "custom":
        p = config.get("Paths", "custom_model_path", fallback="")
        if not p:
            return None, "Custom Vosk model selected but no path is set (Settings > Vosk)."
        if not os.path.isabs(p):
            p = os.path.join(base, p)
        if not os.path.isdir(p):
            return None, f"Custom Vosk model folder not found:\n{p}"
        return p, None
    if preferred in MODEL_INFO:
        d = os.path.join(base, config.get("Models", "model_directory", fallback=MODEL_BASE_DIR),
                         MODEL_INFO[preferred]["extracted_dir_name"])
        if not os.path.isdir(d):
            return None, f"Vosk model '{preferred}' is not downloaded yet.\nUse Settings > Vosk > Download."
        return d, None
    return None, f"Invalid Vosk model type '{preferred}' in config."


def vosk_model_downloaded(key):
    if key not in MODEL_INFO:
        return False
    d = os.path.join(get_base_path(), config.get("Models", "model_directory", fallback=MODEL_BASE_DIR),
                     MODEL_INFO[key]["extracted_dir_name"])
    return os.path.isdir(d)


def sherpa_model_downloaded(key):
    if key not in SHERPA_MODELS:
        return False
    return os.path.isdir(os.path.join(get_base_path(), SHERPA_BASE_DIR,
                                      SHERPA_MODELS[key]["dir_name"]))


def resolve_sherpa_model_dir():
    """Return (model_dir, error). One of the two is always None."""
    choice = config.get("Sherpa", "model", fallback=DEFAULT_SHERPA_MODEL)
    if choice == "custom":
        d = config.get("Sherpa", "custom_model_dir", fallback="")
        if not d:
            return None, "Custom Sherpa model selected but no folder is set (Settings > Streaming)."
        if not os.path.isabs(d):
            d = os.path.join(get_base_path(), d)
        if not os.path.isdir(d):
            return None, f"Custom Sherpa model folder not found:\n{d}"
        return d, None
    if choice in SHERPA_MODELS:
        d = os.path.join(get_base_path(), SHERPA_BASE_DIR, SHERPA_MODELS[choice]["dir_name"])
        if not os.path.isdir(d):
            return None, f"Sherpa model '{choice}' is not downloaded yet.\nUse Settings > Streaming > Download."
        return d, None
    return None, f"Invalid Sherpa model '{choice}' in config."


def find_sherpa_files(model_dir):
    """Locate the transducer files inside a sherpa-onnx model folder."""
    def pick(pattern):
        matches = sorted(glob.glob(os.path.join(model_dir, pattern)))
        if not matches:
            raise RuntimeError(f"No '{pattern}' file found in {model_dir}")
        full = [m for m in matches if "int8" not in os.path.basename(m)]
        return (full or matches)[0]
    tokens = os.path.join(model_dir, "tokens.txt")
    if not os.path.exists(tokens):
        raise RuntimeError(f"No tokens.txt found in {model_dir}")
    return pick("encoder*.onnx"), pick("decoder*.onnx"), pick("joiner*.onnx"), tokens


def speaker_model_path():
    return os.path.join(get_base_path(), SPEAKER_MODEL_DIR, SPEAKER_MODEL_FILE)


def speaker_labels_active():
    return (config.getboolean("Speakers", "enabled", fallback=True)
            and HAVE_SHERPA and os.path.exists(speaker_model_path()))


class SpeakerRegistry:
    """Assigns 'Speaker N' labels by clustering voice embeddings.

    Embeddings come from a sherpa-onnx speaker model; new utterances are
    matched against running per-speaker centroids by cosine similarity.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.centroids = []  # list of [unit-norm embedding, sample count]
        self._extractor = None
        self._extractor_path = None

    def _get_extractor(self):
        path = speaker_model_path()
        if not (HAVE_SHERPA and os.path.exists(path)):
            return None
        if self._extractor is None or self._extractor_path != path:
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=2)
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            self._extractor_path = path
        return self._extractor

    def label(self, audio_16k):
        """Return a 'Speaker N' label for a mono 16 kHz float32 utterance."""
        if audio_16k is None or len(audio_16k) < int(0.8 * 16000):
            return None
        try:
            with self.lock:
                ext = self._get_extractor()
                if ext is None:
                    return None
                stream = ext.create_stream()
                stream.accept_waveform(16000, audio_16k)
                stream.input_finished()
                emb = np.asarray(ext.compute(stream), dtype=np.float32)
                norm = float(np.linalg.norm(emb))
                if not norm:
                    return None
                emb /= norm
                threshold = config.getfloat("Speakers", "similarity_threshold", fallback=0.45)
                max_speakers = config.getint("Speakers", "max_speakers", fallback=8)
                best_idx, best_sim = -1, -1.0
                for i, (centroid, _count) in enumerate(self.centroids):
                    sim = float(np.dot(emb, centroid))
                    if sim > best_sim:
                        best_idx, best_sim = i, sim
                if best_idx >= 0 and (best_sim >= threshold or len(self.centroids) >= max_speakers):
                    centroid, count = self.centroids[best_idx]
                    updated = centroid * count + emb
                    updated /= float(np.linalg.norm(updated)) or 1.0
                    self.centroids[best_idx] = [updated, count + 1]
                    return f"Speaker {best_idx + 1}"
                self.centroids.append([emb, 1])
                return f"Speaker {len(self.centroids)}"
        except Exception as e:
            print(f"Speaker labelling error: {e}")
            return None

    def reset(self):
        with self.lock:
            self.centroids = []


speaker_registry = SpeakerRegistry()


def _download_to_file(url, dest, out_q, label):
    """Stream a URL to a file, posting progress. Raises on failure."""
    if not HAVE_REQUESTS:
        raise RuntimeError("The 'requests' package is required for downloads (pip install requests).")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        last = 0.0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 0.4:
                        last = now
                        if total:
                            out_q.put(("status", f"Downloading {label}: {done/1048576:.0f} / {total/1048576:.0f} MB ({100*done/total:.0f}%)"))
                        else:
                            out_q.put(("status", f"Downloading {label}: {done/1048576:.0f} MB"))


def download_vosk_model(model_key, out_q):
    """Download + extract a Vosk model. Runs on a worker thread; reports via out_q."""
    ok, err = False, None
    label = f"Vosk model '{model_key}'"
    info = MODEL_INFO[model_key]
    target_dir = os.path.join(get_base_path(), config.get("Models", "model_directory", fallback=MODEL_BASE_DIR))
    zip_path = os.path.join(target_dir, f"vosk_model_{model_key}.zip")
    try:
        os.makedirs(target_dir, exist_ok=True)
        _download_to_file(info["url"], zip_path, out_q, label)
        out_q.put(("status", f"Extracting {label}..."))
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        if not vosk_model_downloaded(model_key):
            raise RuntimeError("Archive extracted but the expected model folder was not found.")
        ok = True
    except Exception as e:
        err = str(e)
    finally:
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except OSError:
            pass
    out_q.put(("download_done", (ok, label, err)))


def download_sherpa_model(model_key, out_q):
    """Download + extract a sherpa-onnx streaming model (tar.bz2)."""
    ok, err = False, None
    label = f"Sherpa model '{model_key}'"
    info = SHERPA_MODELS[model_key]
    target_dir = os.path.join(get_base_path(), SHERPA_BASE_DIR)
    archive = os.path.join(target_dir, f"sherpa_{model_key}.tar.bz2")
    try:
        os.makedirs(target_dir, exist_ok=True)
        _download_to_file(info["url"], archive, out_q, label)
        out_q.put(("status", f"Extracting {label}..."))
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(target_dir, filter="data")
        if not sherpa_model_downloaded(model_key):
            raise RuntimeError("Archive extracted but the expected model folder was not found.")
        ok = True
    except Exception as e:
        err = str(e)
    finally:
        try:
            if os.path.exists(archive):
                os.remove(archive)
        except OSError:
            pass
    out_q.put(("download_done", (ok, label, err)))


def download_speaker_model(out_q):
    """Download the speaker-embedding model used for speaker labels."""
    ok, err = False, None
    label = "speaker recognition model"
    dest = speaker_model_path()
    tmp = dest + ".part"
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _download_to_file(SPEAKER_MODEL_URL, tmp, out_q, label)
        os.replace(tmp, dest)
        ok = True
    except Exception as e:
        err = str(e)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
    out_q.put(("download_done", (ok, label, err)))


# --- Engine registry -----------------------------------------------------------
# Every engine is one ENGINES entry. To add a new engine:
#   1. write a prepare_*(status) function returning one of
#        ("vosk", vosk_model)           -> fed by the Vosk streaming loop
#        ("sherpa", online_recognizer)  -> fed by the sherpa streaming loop
#        ("segment", recognize)         -> recognize(float32_audio, rate) -> text,
#                                          fed by the built-in VAD segmenter
#   2. add an ENGINES entry below.
# The engine dropdown, Settings page and validation all build themselves
# from this registry. New downloadable Vosk/Sherpa models can also be added
# without code via a models.json file (see load_model_catalog).

def prepare_vosk(status):
    if not HAVE_VOSK:
        raise RuntimeError("Vosk is not installed (pip install vosk).")
    path, err = resolve_vosk_model_path()
    if err:
        raise RuntimeError(err)
    key = ("vosk", path)
    model = _model_cache.get(key)
    if model is None:
        status(f"Loading Vosk model '{os.path.basename(path)}'...")
        try:
            vosk.SetLogLevel(-1)
        except Exception:
            pass
        model = vosk.Model(path)
        _model_cache[key] = model
    return "vosk", model


def prepare_sherpa(status):
    if not HAVE_SHERPA:
        raise RuntimeError("sherpa-onnx is not installed (pip install sherpa-onnx).")
    model_dir, err = resolve_sherpa_model_dir()
    if err:
        raise RuntimeError(err)
    pause = config.getfloat("Audio", "pause_threshold", fallback=0.7)
    key = ("sherpa", model_dir, round(pause, 2))
    recognizer = _model_cache.get(key)
    if recognizer is None:
        status(f"Loading Sherpa model '{os.path.basename(model_dir)}'...")
        encoder, decoder, joiner, tokens = find_sherpa_files(model_dir)
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens, encoder=encoder, decoder=decoder, joiner=joiner,
            num_threads=2, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=max(0.6, pause),
            rule3_min_utterance_length=300)
        _model_cache[key] = recognizer
    return "sherpa", recognizer


def prepare_moonshine(status):
    if not HAVE_MOONSHINE:
        raise RuntimeError("Moonshine is not installed (pip install useful-moonshine-onnx).")
    size = config.get("Moonshine", "model", fallback=DEFAULT_MOONSHINE_MODEL)
    if size not in MOONSHINE_MODELS:
        size = DEFAULT_MOONSHINE_MODEL
    key = ("moonshine", size)
    cached = _model_cache.get(key)
    if cached is None:
        status(f"Loading {size} (first use downloads the model)...")
        import moonshine_onnx
        model = moonshine_onnx.MoonshineOnnxModel(model_name=size)
        tokenizer = moonshine_onnx.load_tokenizer()
        cached = (model, tokenizer)
        _model_cache[key] = cached
    ms_model, ms_tokenizer = cached

    def recognize(seg, rate):
        audio = resample_to_16k(seg, rate)
        tokens = ms_model.generate(audio[np.newaxis, :].astype(np.float32))
        return ms_tokenizer.decode_batch(tokens)[0].strip()
    return "segment", recognize


def prepare_google_web(status):
    if not HAVE_SR:
        raise RuntimeError("SpeechRecognition is not installed (pip install SpeechRecognition).")
    language = config.get("GoogleWeb", "language", fallback="en-US").strip() or "en-US"
    recognizer = sr_lib.Recognizer()

    def recognize(seg, rate):
        pcm = float_to_int16_bytes(seg)
        audio_data = sr_lib.AudioData(pcm, rate, 2)
        try:
            return recognizer.recognize_google(audio_data, language=language).strip()
        except sr_lib.UnknownValueError:
            return ""
    return "segment", recognize


def prepare_whisper(status):
    backend = config.get("Whisper", "backend", fallback="faster-whisper")
    size = config.get("Whisper", "model_size", fallback=DEFAULT_WHISPER_SIZE).strip() or DEFAULT_WHISPER_SIZE
    lang = config.get("Whisper", "language", fallback="auto").strip().lower()
    lang = None if lang in ("", "auto") else lang

    if backend == "faster-whisper":
        if not HAVE_FASTER_WHISPER:
            raise RuntimeError("faster-whisper is not installed (pip install faster-whisper).")
        # Any name faster-whisper understands is allowed, including Hugging
        # Face CTranslate2 repo ids - this is how future models stay usable.
        device = config.get("Whisper", "device", fallback="auto")
        compute = config.get("Whisper", "compute_type", fallback="auto")
        compute = "default" if compute == "auto" else compute
        vad = config.getboolean("Whisper", "vad_filter", fallback=True)
        key = ("fw", size, device, compute)
        model = _model_cache.get(key)
        if model is None:
            status(f"Loading faster-whisper '{size}' (first use downloads the model)...")
            model = FasterWhisperModel(size, device=device, compute_type=compute)
            if device == "auto":
                # CTranslate2 picks CUDA whenever a GPU is visible, even on
                # machines without the CUDA libraries installed - verify with
                # a tiny warm-up and drop to CPU if that blows up.
                try:
                    warm_segments, _ = model.transcribe(
                        np.zeros(1600, dtype=np.float32), language="en", beam_size=1)
                    list(warm_segments)
                except Exception:
                    status("GPU libraries missing - using CPU for faster-whisper...")
                    model = FasterWhisperModel(size, device="cpu", compute_type=compute)
            _model_cache[key] = model

        def recognize(seg, rate):
            audio = resample_to_16k(seg, rate)
            segments, _info = model.transcribe(
                audio, language=lang, vad_filter=vad, beam_size=1,
                condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segments).strip()
        return "segment", recognize

    if backend == "openai-whisper":
        if not HAVE_OPENAI_WHISPER:
            raise RuntimeError("openai-whisper is not installed (pip install openai-whisper).")
        if size not in OPENAI_WHISPER_SIZES:
            size = DEFAULT_WHISPER_SIZE
        key = ("ow", size)
        model = _model_cache.get(key)
        if model is None:
            status(f"Loading Whisper '{size}' (first use downloads the model)...")
            model = openai_whisper.load_model(size)
            _model_cache[key] = model
        use_fp16 = str(getattr(model, "device", "cpu")) != "cpu"

        def recognize(seg, rate):
            audio = resample_to_16k(seg, rate)
            result = model.transcribe(audio, language=lang, fp16=use_fp16)
            return result.get("text", "").strip()
        return "segment", recognize

    raise RuntimeError(f"Unknown Whisper backend '{backend}'. Check Settings > Whisper.")


def prepare_google_cloud(status):
    if not HAVE_SR:
        raise RuntimeError("SpeechRecognition is not installed (pip install SpeechRecognition google-cloud-speech).")
    creds_path = config.get("Engine", "google_cloud_credentials_json", fallback="")
    if not creds_path:
        raise RuntimeError("Google Cloud selected, but no credentials JSON is set (Settings > Online).")
    if not os.path.isabs(creds_path):
        creds_path = os.path.join(get_base_path(), creds_path)
    if not os.path.exists(creds_path):
        raise RuntimeError(f"Google Cloud credentials file not found:\n{creds_path}")
    with open(creds_path, "r", encoding="utf-8") as f:
        creds = f.read()
    recognizer = sr_lib.Recognizer()

    def recognize(seg, rate):
        pcm = float_to_int16_bytes(seg)
        audio_data = sr_lib.AudioData(pcm, rate, 2)
        try:
            return recognizer.recognize_google_cloud(audio_data, credentials_json=creds).strip()
        except sr_lib.UnknownValueError:
            return ""
    return "segment", recognize


def validate_vosk():
    _path, err = resolve_vosk_model_path()
    return err


def validate_sherpa():
    _path, err = resolve_sherpa_model_dir()
    return err


def validate_whisper():
    backend = config.get("Whisper", "backend", fallback="faster-whisper")
    if backend == "faster-whisper" and not HAVE_FASTER_WHISPER:
        return "faster-whisper is not installed.\n\npip install faster-whisper"
    if backend == "openai-whisper" and not HAVE_OPENAI_WHISPER:
        return "openai-whisper is not installed.\n\npip install openai-whisper"
    return None


def validate_google_cloud():
    creds = config.get("Engine", "google_cloud_credentials_json", fallback="")
    if not creds:
        return "Google Cloud needs a credentials JSON file (Settings > Online)."
    if not os.path.isabs(creds):
        creds = os.path.join(get_base_path(), creds)
    if not os.path.exists(creds):
        return f"Google Cloud credentials file not found:\n{creds}"
    return None


def _validate_none():
    return None


ENGINES = {
    "vosk": {
        "label": "Vosk",
        "title": "Vosk (offline)",
        "desc": "True streaming - words appear live. Light on resources.",
        "hint": "Vosk: streaming with live partial results.",
        "available": lambda: HAVE_VOSK,
        "install": "pip install vosk",
        "validate": validate_vosk,
        "prepare": prepare_vosk,
    },
    "sherpa": {
        "label": "Sherpa (streaming)",
        "title": "Sherpa streaming (offline)",
        "desc": "Modern streaming Zipformer models - live partials like Vosk, better accuracy.",
        "hint": "Sherpa: streaming with live partials; newer models than Vosk.",
        "available": lambda: HAVE_SHERPA,
        "install": "pip install sherpa-onnx",
        "validate": validate_sherpa,
        "prepare": prepare_sherpa,
    },
    "whisper": {
        "label": "Whisper",
        "title": "Whisper (offline)",
        "desc": "Best accuracy, phrase by phrase. faster-whisper backend recommended.",
        "hint": "Whisper: best accuracy; transcribes phrase by phrase.",
        "available": lambda: HAVE_FASTER_WHISPER or HAVE_OPENAI_WHISPER,
        "install": "pip install faster-whisper",
        "validate": validate_whisper,
        "prepare": prepare_whisper,
    },
    "moonshine": {
        "label": "Moonshine",
        "title": "Moonshine (offline)",
        "desc": "New fast English model built for live captions; phrase by phrase.",
        "hint": "Moonshine: fast local English model; phrase by phrase.",
        "available": lambda: HAVE_MOONSHINE,
        "install": "pip install useful-moonshine-onnx",
        "validate": _validate_none,
        "prepare": prepare_moonshine,
    },
    "google_web": {
        "label": "Google Web (free)",
        "title": "Google Web Speech (online)",
        "desc": "Free, no key or setup needed. Sends audio to Google.",
        "hint": "Google Web Speech: free online, no key needed.",
        "available": lambda: HAVE_SR,
        "install": "pip install SpeechRecognition",
        "validate": _validate_none,
        "prepare": prepare_google_web,
    },
    "google_cloud": {
        "label": "Google Cloud",
        "title": "Google Cloud (online)",
        "desc": "Requires a credentials JSON file. Sends audio to Google.",
        "hint": "Google Cloud: online; needs credentials in Settings.",
        "available": lambda: HAVE_SR,
        "install": "pip install SpeechRecognition google-cloud-speech",
        "validate": validate_google_cloud,
        "prepare": prepare_google_cloud,
    },
}

ENGINE_LABELS = {key: spec["label"] for key, spec in ENGINES.items()}


# --- Transcription session ----------------------------------------------------
class TranscriptionSession:
    """Owns the audio stream and worker threads for one start/stop cycle."""

    def __init__(self, out_q, sources):
        self.out = out_q
        # sources: list of (kind, index, name) descriptors. The first is the
        # clock master; any others are resampled and summed into it.
        self.sources = sources if isinstance(sources, list) else [sources]
        self.stop_event = threading.Event()
        self.audio_q = queue.Queue()
        self.seg_q = queue.Queue()
        self.thread = None
        self.recog_thread = None
        self.samplerate = 16000
        self.engine = "vosk"
        self._sd_streams = []
        self._pa = None
        self._pa_streams = []
        self._mix_lock = threading.Lock()
        self._mix_bufs = []  # one ring buffer per secondary source

    def start(self):
        self.thread = threading.Thread(target=self._run, name="TranscribeSession", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    # -- audio capture --
    @staticmethod
    def _to_mono(indata):
        if indata.ndim > 1 and indata.shape[1] > 1:
            return indata.mean(axis=1)
        return indata[:, 0] if indata.ndim > 1 else indata

    def _resolve_sd_device(self, index, name):
        """Return (index, info) for the device, re-scanning by name if the
        index went stale (device lists renumber when drivers restart)."""
        try:
            info = sd.query_devices(index)
            if info.get("name") == name and info.get("max_input_channels", 0) > 0:
                return index, info
        except Exception:
            pass
        refresh_portaudio()
        candidates = []
        try:
            apis = sd.query_hostapis()
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0 and d.get("name") == name:
                    candidates.append((i, d, apis[d["hostapi"]]["name"]))
        except Exception:
            pass
        for i, d, api in candidates:
            if "WASAPI" in api:
                return i, d
        if candidates:
            return candidates[0][0], candidates[0][1]
        raise RuntimeError(f"Audio source '{name}' is no longer available.\n"
                           "Press the refresh button and pick a source again.")

    def _open_sd_stream(self, index, name, callback):
        """Open a sounddevice input, walking a fallback cascade of sample
        rates and channel counts. Returns (samplerate, device_info).

        If the whole cascade fails, PortAudio is re-initialized and the
        cascade retried once: Windows device state gets wedged (GoXLR
        utility restarts, the loopback library's own PortAudio instance)
        and a fresh init reliably clears it.
        """
        last_err = None
        for attempt in range(2):
            index, info = self._resolve_sd_device(index, name)
            sr = int(info.get("default_samplerate") or 0) or 48000
            max_ch = max(1, int(info.get("max_input_channels") or 1))
            is_wasapi = False
            try:
                is_wasapi = "WASAPI" in sd.query_hostapis(info["hostapi"])["name"]
            except Exception:
                pass
            tried = set()
            for rate, ch in [(sr, 1), (sr, min(2, max_ch)),
                             (48000, min(2, max_ch)), (44100, min(2, max_ch))]:
                if (rate, ch) in tried:
                    continue
                tried.add((rate, ch))
                try:
                    extra = sd.WasapiSettings(auto_convert=True) if is_wasapi else None
                    stream = sd.InputStream(
                        samplerate=rate, device=index, channels=ch, dtype="float32",
                        blocksize=max(256, int(rate * 0.1)), callback=callback,
                        extra_settings=extra)
                    stream.start()
                    self._sd_streams.append(stream)
                    return rate, info
                except Exception as e:
                    last_err = e
            if attempt == 0:
                refresh_portaudio()
        raise RuntimeError(f"Could not open audio source '{name}':\n{last_err}\n\n"
                           "Press the refresh button and try again, or pick another source.")

    def _open_loopback_stream(self, index, name, on_chunk):
        """Open a WASAPI loopback capture via pyaudiowpatch. on_chunk receives
        (mono_float32, samplerate). Returns (samplerate, device_info)."""
        if not HAVE_LOOPBACK:
            raise RuntimeError("System-audio capture needs the PyAudioWPatch package\n(pip install PyAudioWPatch).")
        if self._pa is None:
            self._pa = pyaudio_patch.PyAudio()
        dev = None
        try:
            cand = self._pa.get_device_info_by_index(index)
            if cand.get("name") == name and cand.get("isLoopbackDevice"):
                dev = cand
        except Exception:
            pass
        if dev is None:  # index went stale - find it by name
            for d in self._pa.get_loopback_device_info_generator():
                if d["name"] == name:
                    dev = d
                    break
        if dev is None:
            raise RuntimeError(f"System-audio source '{name}' is no longer available.\n"
                               "Press the refresh button and pick a source again.")
        rate = int(dev.get("defaultSampleRate") or 0) or 48000
        ch = max(1, int(dev.get("maxInputChannels") or 2))

        def cb(in_data, frame_count, time_info, status_flags):
            if not self.stop_event.is_set():
                pcm = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    pcm = pcm.reshape(-1, ch).mean(axis=1)
                on_chunk(pcm, rate)
            return (None, pyaudio_patch.paContinue)

        stream = self._pa.open(format=pyaudio_patch.paInt16, channels=ch, rate=rate,
                               frames_per_buffer=max(256, int(rate * 0.1)), input=True,
                               input_device_index=int(dev["index"]), stream_callback=cb)
        self._pa_streams.append(stream)
        return rate, dev

    def _master_push(self, mono):
        """Sum every secondary ring buffer into this master chunk, then enqueue.

        Secondary buffers are already resampled to the master rate, so this is
        a sample-aligned add of whatever each one has buffered so far.
        """
        mono = np.array(mono, dtype=np.float32, copy=True)
        if self._mix_bufs:
            with self._mix_lock:
                for i, buf in enumerate(self._mix_bufs):
                    take = min(len(mono), len(buf))
                    if take:
                        mono[:take] += buf[:take]
                        self._mix_bufs[i] = buf[take:]
            np.clip(mono, -1.0, 1.0, out=mono)
        self.audio_q.put(mono)

    def _make_secondary_sink(self, slot, master_rate):
        max_buf = master_rate * 2  # cap backlog at 2 seconds

        def sink(pcm, rate):
            pcm = resample_audio(pcm, rate, master_rate)
            with self._mix_lock:
                self._mix_bufs[slot] = np.concatenate((self._mix_bufs[slot], pcm))
                if len(self._mix_bufs[slot]) > max_buf:
                    self._mix_bufs[slot] = self._mix_bufs[slot][-max_buf:]
        return sink

    @staticmethod
    def _clean_loop_name(name):
        return name.replace(" [Loopback]", "")

    def _open_master(self, desc):
        """Open the clock-master source; sets self.samplerate; returns a label."""
        kind, index, name = desc
        if kind == "sd":
            def cb(indata, frames, time_info, status_flags):
                if not self.stop_event.is_set():
                    self._master_push(self._to_mono(indata))
            rate, info = self._open_sd_stream(index, name, cb)
            self.samplerate = rate
            return info["name"]
        if kind == "loopback":
            rate, dev = self._open_loopback_stream(
                index, name, lambda pcm, _r: self._master_push(pcm))
            self.samplerate = rate
            return f"System audio: {self._clean_loop_name(dev['name'])}"
        raise RuntimeError(f"Unknown audio source type '{kind}'.")

    def _open_secondary(self, desc, sink):
        """Open an extra source feeding sink(mono_float32, native_rate)."""
        kind, index, name = desc
        if kind == "sd":
            # The sd callback delivers audio at this device's own rate; seed
            # the box with the master rate so the rare chunk that fires before
            # the open returns is still resampled sensibly.
            rate_box = [self.samplerate]

            def cb(indata, frames, time_info, status_flags):
                if not self.stop_event.is_set():
                    sink(np.asarray(self._to_mono(indata), dtype=np.float32), rate_box[0])
            rate, info = self._open_sd_stream(index, name, cb)
            rate_box[0] = rate
            return info["name"]
        if kind == "loopback":
            _rate, dev = self._open_loopback_stream(index, name, sink)
            return f"System audio: {self._clean_loop_name(dev['name'])}"
        raise RuntimeError(f"Unknown audio source type '{kind}'.")

    def _expand_default_mix(self):
        """Resolve the legacy 'default mic + default system audio' shortcut
        into two concrete descriptors."""
        if not HAVE_LOOPBACK:
            raise RuntimeError("Mic + System audio needs the PyAudioWPatch package\n(pip install PyAudioWPatch).")
        try:
            mic_index = sd.default.device[0]
            mic_name = sd.query_devices(mic_index)["name"]
        except Exception:
            raise RuntimeError("No default microphone found for the mic + system mix.")
        if self._pa is None:
            self._pa = pyaudio_patch.PyAudio()
        loop = None
        try:
            wasapi = self._pa.get_host_api_info_by_type(pyaudio_patch.paWASAPI)
            spk = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            for d in self._pa.get_loopback_device_info_generator():
                if spk["name"] in d["name"]:
                    loop = d
                    break
        except Exception:
            pass
        if loop is None:
            loop = next(self._pa.get_loopback_device_info_generator(), None)
        if loop is None:
            raise RuntimeError("No system-audio loopback device found.")
        return [("sd", mic_index, mic_name),
                ("loopback", int(loop["index"]), loop["name"])]

    def _start_capture(self):
        """Open all selected sources (mixing extras into the first). Sets
        self.samplerate and returns a display name for the status bar."""
        # Flatten, expanding any 'mix' shortcut, then drop duplicate sources.
        flat = []
        for d in self.sources:
            if d[0] == "mix":
                flat.extend(self._expand_default_mix())
            else:
                flat.append(d)
        seen, sources = set(), []
        for d in flat:
            key = (d[0], d[1], d[2])
            if key not in seen:
                seen.add(key)
                sources.append(d)
        if not sources:
            raise RuntimeError("No audio source selected.")

        names = [self._open_master(sources[0])]
        for desc in sources[1:]:
            self._mix_bufs.append(np.zeros(0, dtype=np.float32))
            sink = self._make_secondary_sink(len(self._mix_bufs) - 1, self.samplerate)
            names.append(self._open_secondary(desc, sink))
        return " + ".join(names)

    def _stop_capture(self):
        for s in self._sd_streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._sd_streams = []
        for s in self._pa_streams:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                pass
        self._pa_streams = []
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    # -- main session thread --
    def _run(self):
        log_file = None
        try:
            self.engine = config.get("Engine", "type", fallback="vosk")
            mode, payload = self._prepare_engine(self.engine)
            if self.stop_event.is_set():
                return

            source_name = self._start_capture()

            log_file = self._open_log()
            self.out.put(("started", None))
            self.out.put(("status", f"Listening on: {source_name}"))

            if (config.getboolean("Speakers", "enabled", fallback=True)
                    and HAVE_SHERPA and not os.path.exists(speaker_model_path())):
                self.out.put(("error", "Speaker labels are enabled but the speaker model isn't downloaded (Settings > Speakers)."))

            if mode == "vosk":
                rec = vosk.KaldiRecognizer(payload, self.samplerate)
                self._vosk_loop(rec, log_file)
            elif mode == "sherpa":
                self._sherpa_loop(payload, log_file)
            else:
                self.recog_thread = threading.Thread(
                    target=self._recognize_loop, args=(payload, log_file),
                    name="Recognizer", daemon=True)
                self.recog_thread.start()
                self._vad_loop()
                self.recog_thread.join(timeout=20)
        except sd.PortAudioError as e:
            self.out.put(("fatal", f"Audio device error: {e}\n\nTry another audio source or click the refresh button."))
        except Exception as e:
            self.out.put(("fatal", f"{type(e).__name__}: {e}"))
        finally:
            self._stop_capture()
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass
            self.stop_event.set()
            self.out.put(("stopped", None))

    # -- engine setup --
    def _prepare_engine(self, engine):
        """Look the engine up in the ENGINES registry and build its recognizer."""
        spec = ENGINES.get(engine)
        if spec is None:
            raise RuntimeError(f"Unsupported engine '{engine}' in config.")
        return spec["prepare"](lambda msg: self.out.put(("status", msg)))

    # -- logging --
    def _open_log(self):
        if not config.getboolean("Settings", "enable_logging", fallback=True):
            return None
        path = config.get("Audio", "log_file", fallback="live_transcription.log")
        if not os.path.isabs(path):
            path = os.path.join(get_base_path(), path)
        try:
            log_dir = os.path.dirname(path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            f = open(path, "a", buffering=1, encoding="utf-8")
            self.out.put(("status", f"Logging to {path}"))
            return f
        except OSError as e:
            self.out.put(("error", f"Could not open log file ({e}); logging disabled for this run."))
            return None

    def _emit_final(self, text, log_file, speaker=None):
        ts = datetime.now().strftime("%H:%M:%S")
        self.out.put(("final", (ts, text, speaker)))
        if log_file:
            try:
                prefix = f"{speaker}: " if speaker else ""
                log_file.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {prefix}{text}\n")
            except Exception:
                pass

    def _label_speaker(self, chunks):
        """Label the speaker of an utterance held as a list of float32 chunks."""
        if not chunks:
            return None
        try:
            audio = np.concatenate(chunks)
            return speaker_registry.label(resample_to_16k(audio, self.samplerate))
        except Exception:
            return None

    # -- Vosk streaming loop --
    def _vosk_loop(self, rec, log_file):
        last_partial = ""
        collect = speaker_labels_active()
        utt, utt_len = [], 0
        max_keep = int(self.samplerate * 30)
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if collect:
                utt.append(chunk)
                utt_len += len(chunk)
                while utt_len > max_keep and utt:
                    utt_len -= len(utt.pop(0))
            data = float_to_int16_bytes(chunk)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    speaker = self._label_speaker(utt) if collect else None
                    self._emit_final(text, log_file, speaker)
                utt, utt_len = [], 0
                if last_partial:
                    last_partial = ""
                    self.out.put(("partial", ""))
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial != last_partial:
                    last_partial = partial
                    self.out.put(("partial", partial))
        try:
            text = json.loads(rec.FinalResult()).get("text", "").strip()
            if text:
                self._emit_final(text, log_file, self._label_speaker(utt) if collect else None)
        except Exception:
            pass

    # -- sherpa-onnx streaming loop --
    def _sherpa_loop(self, recognizer, log_file):
        stream = recognizer.create_stream()
        last_partial = ""
        collect = speaker_labels_active()
        utt, utt_len = [], 0
        max_keep = int(self.samplerate * 30)
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if collect:
                utt.append(chunk)
                utt_len += len(chunk)
                while utt_len > max_keep and utt:
                    utt_len -= len(utt.pop(0))
            stream.accept_waveform(self.samplerate, chunk)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            text = recognizer.get_result(stream).strip()
            if recognizer.is_endpoint(stream):
                if text:
                    speaker = self._label_speaker(utt) if collect else None
                    self._emit_final(text, log_file, speaker)
                recognizer.reset(stream)
                utt, utt_len = [], 0
                if last_partial:
                    last_partial = ""
                    self.out.put(("partial", ""))
            elif text != last_partial:
                last_partial = text
                self.out.put(("partial", text))
        text = recognizer.get_result(stream).strip()
        if text:
            self._emit_final(text, log_file, self._label_speaker(utt) if collect else None)

    # -- segmenting (energy VAD) loop for Whisper / Google --
    def _vad_loop(self):
        sr_hz = self.samplerate
        pause = config.getfloat("Audio", "pause_threshold", fallback=0.7)
        max_phrase = config.getfloat("Audio", "max_phrase_sec", fallback=12.0)
        threshold = get_energy_threshold()
        calibrating = threshold is None
        noise_levels = []
        cal_time = 0.0
        pre_roll = deque()
        pre_roll_dur = 0.0
        voiced = []
        voiced_dur = 0.0
        speech_dur = 0.0
        silence_run = 0.0

        if calibrating:
            self.out.put(("status", "Calibrating noise floor (stay quiet for a moment)..."))

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.25)
            except queue.Empty:
                continue
            d = len(chunk) / sr_hz
            level = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0

            if calibrating:
                noise_levels.append(level)
                cal_time += d
                if cal_time >= 0.6:
                    threshold = max(float(np.median(noise_levels)) * 3.5, 0.006)
                    calibrating = False
                    self.out.put(("status", "Listening..."))
                continue

            if level >= threshold:
                if not voiced:
                    voiced.extend(pre_roll)
                    voiced_dur += pre_roll_dur
                    pre_roll.clear()
                    pre_roll_dur = 0.0
                    self.out.put(("status", "Hearing speech..."))
                voiced.append(chunk)
                voiced_dur += d
                speech_dur += d
                silence_run = 0.0
            elif voiced:
                voiced.append(chunk)
                voiced_dur += d
                silence_run += d
                if silence_run >= pause:
                    self._dispatch(voiced, speech_dur)
                    voiced, voiced_dur, speech_dur, silence_run = [], 0.0, 0.0, 0.0
            else:
                pre_roll.append(chunk)
                pre_roll_dur += d
                while pre_roll_dur > 0.3 and pre_roll:
                    old = pre_roll.popleft()
                    pre_roll_dur -= len(old) / sr_hz

            if voiced and voiced_dur >= max_phrase:
                self._dispatch(voiced, speech_dur)
                voiced, voiced_dur, speech_dur, silence_run = [], 0.0, 0.0, 0.0

        if voiced:
            self._dispatch(voiced, speech_dur)

    def _dispatch(self, chunks, speech_dur):
        if speech_dur < 0.3:
            return
        self.seg_q.put(np.concatenate(chunks))

    def _recognize_loop(self, recognize, log_file):
        while True:
            try:
                seg = self.seg_q.get(timeout=0.25)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            try:
                self.out.put(("status", "Transcribing..."))
                text = recognize(seg, self.samplerate)
            except Exception as e:
                self.out.put(("error", f"Recognition failed: {e}"))
                continue
            if text:
                speaker = None
                if speaker_labels_active():
                    speaker = speaker_registry.label(resample_to_16k(seg, self.samplerate))
                self._emit_final(text, log_file, speaker)
            if not self.stop_event.is_set():
                self.out.put(("status", "Listening..."))


# --- Windows dark title bar -----------------------------------------------------
def set_titlebar_dark(window, dark):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (20; 19 pre-20H1)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
        # Nudge the window so DWM repaints the title bar immediately.
        try:
            alpha = window.attributes("-alpha")
            window.attributes("-alpha", max(0.1, float(alpha) - 0.01))
            window.after(20, lambda: window.attributes("-alpha", alpha))
        except Exception:
            pass
    except Exception:
        pass


# --- GUI -------------------------------------------------------------------------
class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.session = None
        self.settings_dialog = None
        self.overlay_on = False
        self._restore_topmost = False
        self.palette = DARK

        root.title(APP_TITLE)
        root.geometry("880x560")
        root.minsize(680, 380)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.bind("<Escape>", lambda e: self.exit_overlay())

        self.style = ttk.Style(root)

        base_family = "Segoe UI" if "Segoe UI" in tkfont.families(root) else "TkDefaultFont"
        self.base_font = (base_family, 10)
        self.transcript_font = tkfont.Font(
            family=base_family,
            size=config.getint("Settings", "font_size", fallback=11))
        self.partial_font = tkfont.Font(
            family=base_family,
            size=config.getint("Settings", "font_size", fallback=11),
            slant="italic")
        self.speaker_font = tkfont.Font(
            family=base_family,
            size=config.getint("Settings", "font_size", fallback=11),
            weight="bold")

        # ---- top bar (two rows) ----
        self.topbar = ttk.Frame(root, padding=(10, 8, 10, 4))
        self.topbar.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(self.topbar)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Source:").pack(side=tk.LEFT)
        self.devices = list_audio_devices()
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(row1, textvariable=self.device_var,
                                         values=list(self.devices.keys()),
                                         state="readonly", width=40)
        self.device_combo.pack(side=tk.LEFT, padx=(6, 2))
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        ttk.Label(row1, text="+ Mix with:").pack(side=tk.LEFT, padx=(8, 0))
        self.mix_var = tk.StringVar()
        self.mix_combo = ttk.Combobox(row1, textvariable=self.mix_var,
                                      values=[NO_MIX_LABEL] + list(self.devices.keys()),
                                      state="readonly", width=34)
        self.mix_combo.pack(side=tk.LEFT, padx=(6, 2))
        self.mix_combo.bind("<<ComboboxSelected>>", self.on_mix_change)
        self.refresh_btn = ttk.Button(row1, text="⟳", width=3, command=self.refresh_devices)
        self.refresh_btn.pack(side=tk.LEFT)

        row2 = ttk.Frame(self.topbar)
        row2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row2, text="Engine:").pack(side=tk.LEFT)
        self.engine_var = tk.StringVar()
        self._engine_display_to_key = {ENGINE_LABELS[k]: k for k in available_engines()}
        self.engine_combo = ttk.Combobox(row2, textvariable=self.engine_var,
                                         values=list(self._engine_display_to_key.keys()),
                                         state="readonly", width=17)
        self.engine_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.engine_combo.bind("<<ComboboxSelected>>", self.on_engine_change)

        self.start_btn = ttk.Button(row2, text="▶ Start", style="Accent.TButton",
                                    command=self.start_transcription)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(row2, text="■ Stop", command=self.stop_transcription,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.settings_btn = ttk.Button(row2, text="Settings", command=self.open_settings)
        self.settings_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.save_btn = ttk.Button(row2, text="Save transcript...", command=self.save_transcript)
        self.save_btn.pack(side=tk.RIGHT)
        self.clear_btn = ttk.Button(row2, text="Clear", command=self.clear_transcript)
        self.clear_btn.pack(side=tk.RIGHT, padx=(0, 6))

        row3 = ttk.Frame(self.topbar)
        row3.pack(fill=tk.X, pady=(6, 0))
        self.theme_btn = ttk.Button(row3, text="", width=10, command=self.toggle_theme)
        self.theme_btn.pack(side=tk.LEFT)

        self.pin_var = tk.BooleanVar(value=config.getboolean("Settings", "always_on_top", fallback=False))
        self.pin_check = ttk.Checkbutton(row3, text="\U0001F4CC Always on top",
                                         variable=self.pin_var, command=self.apply_pin)
        self.pin_check.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(row3, text="Opacity:").pack(side=tk.LEFT, padx=(14, 4))
        self.opacity_var = tk.DoubleVar(value=config.getfloat("Settings", "opacity", fallback=100))
        self.opacity_scale = ttk.Scale(row3, from_=40, to=100, length=120,
                                       variable=self.opacity_var, command=self.on_opacity_change)
        self.opacity_scale.pack(side=tk.LEFT)

        self.overlay_btn = ttk.Button(row3, text="Overlay mode", command=self.toggle_overlay)
        self.overlay_btn.pack(side=tk.LEFT, padx=(14, 0))

        # ---- transcript body ----
        self.body = ttk.Frame(root, padding=(10, 4, 10, 0))
        self.body.pack(fill=tk.BOTH, expand=True)
        self.body.rowconfigure(0, weight=1)
        self.body.columnconfigure(0, weight=1)

        self.txt = tk.Text(self.body, wrap=tk.WORD, state=tk.DISABLED,
                           font=self.transcript_font, relief=tk.FLAT,
                           borderwidth=0, highlightthickness=1, padx=8, pady=6)
        self.txt.grid(row=0, column=0, sticky="nsew")
        self.scroll = ttk.Scrollbar(self.body, orient=tk.VERTICAL, command=self.txt.yview)
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=self.scroll.set)

        self.partial_var = tk.StringVar(value="")
        self.partial_label = tk.Label(self.body, textvariable=self.partial_var,
                                      anchor="w", justify=tk.LEFT, font=self.partial_font)
        self.partial_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 4))

        # ---- status bar ----
        self.status_var = tk.StringVar(value="Idle. Pick an audio source and press Start.")
        self.status_bar = ttk.Label(root, textvariable=self.status_var,
                                    style="Status.TLabel", anchor=tk.W, padding=(10, 4))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ---- initial state ----
        self._select_initial_device()
        self._select_initial_engine()
        self.apply_theme(config.get("Settings", "theme", fallback="dark"))
        self.apply_pin()
        self.on_opacity_change(self.opacity_var.get())
        if not available_engines():
            self.status_var.set("No transcription engines installed. See README / requirements.txt.")
            self.start_btn.config(state=tk.DISABLED)
        self.root.after(80, self.pump_queue)

    # ---------- device & engine selection ----------
    def _select_initial_device(self):
        if not self.devices:
            self.device_var.set("No audio input devices found")
            self.device_combo.config(state=tk.DISABLED)
            self.mix_combo.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.DISABLED)
            return
        saved = config.get("Audio", "audio_source_name", fallback="")
        names = list(self.devices.keys())
        if saved in names:
            self.device_var.set(saved)
        else:
            default = [n for n in names if "(default)" in n]
            self.device_var.set(default[0] if default else names[0])
        config.set("Audio", "audio_source_name", self.device_var.get())
        saved_mix = config.get("Audio", "mix_source_name", fallback="")
        self.mix_var.set(saved_mix if saved_mix in names else NO_MIX_LABEL)

    def _select_initial_engine(self):
        engines = available_engines()
        if not engines:
            self.engine_var.set("None installed")
            self.engine_combo.config(state=tk.DISABLED)
            return
        current = config.get("Engine", "type", fallback="vosk")
        if current not in engines:
            current = engines[0]
            config.set("Engine", "type", current)
        self.engine_var.set(ENGINE_LABELS[current])

    def on_device_change(self, event=None):
        name = self.device_var.get()
        if name in self.devices:
            config.set("Audio", "audio_source_name", name)

    def on_mix_change(self, event=None):
        name = self.mix_var.get()
        config.set("Audio", "mix_source_name", name if name in self.devices else "")

    def on_engine_change(self, event=None):
        key = self._engine_display_to_key.get(self.engine_var.get())
        if key:
            config.set("Engine", "type", key)
            self.status_var.set(ENGINES[key]["hint"])

    def refresh_devices(self):
        previous = self.device_var.get()
        refresh_portaudio()  # pick up devices added/removed since startup
        previous_mix = self.mix_var.get()
        self.devices = list_audio_devices()
        names = list(self.devices.keys())
        self.device_combo.config(values=names, state="readonly" if names else tk.DISABLED)
        self.mix_combo.config(values=[NO_MIX_LABEL] + names,
                              state="readonly" if names else tk.DISABLED)
        if not names:
            self.device_var.set("No audio input devices found")
            self.start_btn.config(state=tk.DISABLED)
            return
        if previous in names:
            self.device_var.set(previous)
        else:
            default = [n for n in names if "(default)" in n]
            self.device_var.set(default[0] if default else names[0])
        config.set("Audio", "audio_source_name", self.device_var.get())
        self.mix_var.set(previous_mix if previous_mix in names else NO_MIX_LABEL)
        config.set("Audio", "mix_source_name",
                   self.mix_var.get() if self.mix_var.get() in names else "")
        if self.session is None or not self.session.is_running():
            self.start_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Found {len(names)} audio source(s).")

    # ---------- theming ----------
    def apply_theme(self, mode):
        p = DARK if mode == "dark" else LIGHT
        self.palette = p
        config.set("Settings", "theme", mode)
        s = self.style
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass

        self.root.configure(bg=p["bg"])
        s.configure(".", background=p["bg"], foreground=p["fg"],
                    bordercolor=p["border"], focuscolor=p["accent"],
                    font=self.base_font)
        s.configure("TFrame", background=p["bg"])
        s.configure("TLabel", background=p["bg"], foreground=p["fg"])
        s.configure("Muted.TLabel", background=p["bg"], foreground=p["muted"])
        s.configure("Status.TLabel", background=p["surface"], foreground=p["muted"])

        s.configure("TButton", background=p["control"], foreground=p["fg"],
                    bordercolor=p["border"], padding=(10, 5), relief=tk.FLAT)
        s.map("TButton",
              background=[("disabled", p["bg"]), ("pressed", p["control_hover"]),
                          ("active", p["control_hover"])],
              foreground=[("disabled", p["muted"])])
        s.configure("Accent.TButton", background=p["accent"], foreground=p["accent_fg"])
        s.map("Accent.TButton",
              background=[("disabled", p["control"]), ("pressed", p["accent_hover"]),
                          ("active", p["accent_hover"])],
              foreground=[("disabled", p["muted"])])

        s.configure("TCheckbutton", background=p["bg"], foreground=p["fg"],
                    indicatorcolor=p["control"])
        s.map("TCheckbutton",
              background=[("active", p["bg"])],
              indicatorcolor=[("selected", p["accent"])],
              foreground=[("disabled", p["muted"])])
        s.configure("TRadiobutton", background=p["bg"], foreground=p["fg"],
                    indicatorcolor=p["control"])
        s.map("TRadiobutton",
              background=[("active", p["bg"])],
              indicatorcolor=[("selected", p["accent"])],
              foreground=[("disabled", p["muted"])])

        s.configure("TCombobox", fieldbackground=p["control"], background=p["control"],
                    foreground=p["fg"], arrowcolor=p["fg"], bordercolor=p["border"],
                    lightcolor=p["control"], darkcolor=p["control"])
        s.map("TCombobox",
              fieldbackground=[("readonly", p["control"]), ("disabled", p["bg"])],
              foreground=[("disabled", p["muted"])],
              selectbackground=[("readonly", p["control"])],
              selectforeground=[("readonly", p["fg"])])
        s.configure("TEntry", fieldbackground=p["control"], foreground=p["fg"],
                    insertcolor=p["fg"], bordercolor=p["border"])
        s.configure("TSpinbox", fieldbackground=p["control"], foreground=p["fg"],
                    insertcolor=p["fg"], arrowcolor=p["fg"], bordercolor=p["border"])

        s.configure("TLabelframe", background=p["bg"], bordercolor=p["border"])
        s.configure("TLabelframe.Label", background=p["bg"], foreground=p["muted"])
        s.configure("TNotebook", background=p["bg"], bordercolor=p["border"])
        s.configure("TNotebook.Tab", background=p["control"], foreground=p["fg"],
                    padding=(12, 6))
        s.map("TNotebook.Tab",
              background=[("selected", p["bg"])],
              foreground=[("selected", p["accent"])])
        s.configure("Horizontal.TScale", background=p["bg"], troughcolor=p["control"])
        s.configure("Vertical.TScrollbar", background=p["control"], troughcolor=p["bg"],
                    bordercolor=p["bg"], arrowcolor=p["muted"])
        s.map("Vertical.TScrollbar", background=[("active", p["control_hover"])])

        # Combobox dropdown list (plain tk listbox inside the popdown).
        self.root.option_add("*TCombobox*Listbox.background", p["surface"])
        self.root.option_add("*TCombobox*Listbox.foreground", p["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", p["sel_bg"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", p["fg"])

        self.txt.configure(bg=p["text_bg"], fg=p["text_fg"], insertbackground=p["fg"],
                           selectbackground=p["sel_bg"], selectforeground=p["fg"],
                           highlightbackground=p["border"], highlightcolor=p["border"])
        self.txt.tag_configure("ts", foreground=p["muted"])
        self.txt.tag_configure("info", foreground=p["accent"], font=self.partial_font)
        self.txt.tag_configure("warn", foreground=p["danger"])
        for i, colour in enumerate(p["speakers"]):
            self.txt.tag_configure(f"spk{i}", foreground=colour, font=self.speaker_font)
        self.partial_label.configure(bg=p["bg"], fg=p["muted"])

        self.theme_btn.config(text="☀ Light" if mode == "dark" else "\U0001F319 Dark")
        set_titlebar_dark(self.root, mode == "dark")
        if self.settings_dialog is not None and self.settings_dialog.winfo_exists():
            self.settings_dialog.configure(bg=p["bg"])
            set_titlebar_dark(self.settings_dialog, mode == "dark")

    def toggle_theme(self):
        mode = "light" if config.get("Settings", "theme", fallback="dark") == "dark" else "dark"
        self.apply_theme(mode)
        save_config()

    def set_font_size(self, size):
        config.set("Settings", "font_size", str(size))
        self.transcript_font.configure(size=size)
        self.partial_font.configure(size=size)
        self.speaker_font.configure(size=size)

    # ---------- transparency / pin / overlay ----------
    def on_opacity_change(self, value):
        try:
            v = max(40.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            v = 100.0
        if not self.overlay_on:
            self.root.attributes("-alpha", v / 100.0)
        config.set("Settings", "opacity", f"{v:.0f}")

    def apply_pin(self):
        on = bool(self.pin_var.get())
        self.root.attributes("-topmost", on)
        config.set("Settings", "always_on_top", str(on))

    def toggle_overlay(self):
        if self.overlay_on:
            self.exit_overlay()
            return
        self.overlay_on = True
        self._restore_topmost = bool(self.pin_var.get())
        self.topbar.pack_forget()
        self.status_bar.pack_forget()
        self.root.attributes("-topmost", True)
        overlay_alpha = config.getfloat("Settings", "overlay_opacity", fallback=85) / 100.0
        self.root.attributes("-alpha", max(0.3, min(1.0, overlay_alpha)))
        self.partial_var.set("Overlay mode - press Esc to restore controls")

    def exit_overlay(self):
        if not self.overlay_on:
            return
        self.overlay_on = False
        self.topbar.pack(side=tk.TOP, fill=tk.X, before=self.body)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.attributes("-topmost", self._restore_topmost)
        self.on_opacity_change(self.opacity_var.get())
        if self.partial_var.get().startswith("Overlay mode"):
            self.partial_var.set("")

    # ---------- transcript helpers ----------
    def append_final(self, ts, text, speaker=None):
        self.txt.configure(state=tk.NORMAL)
        self.txt.insert(tk.END, f"[{ts}] ", "ts")
        if speaker:
            try:
                idx = (int(speaker.split()[-1]) - 1) % len(self.palette["speakers"])
            except (ValueError, IndexError):
                idx = 0
            self.txt.insert(tk.END, f"{speaker}: ", f"spk{idx}")
        self.txt.insert(tk.END, text + "\n")
        self.txt.see(tk.END)
        self.txt.configure(state=tk.DISABLED)

    def append_info(self, text, tag="info"):
        self.txt.configure(state=tk.NORMAL)
        self.txt.insert(tk.END, text + "\n", tag)
        self.txt.see(tk.END)
        self.txt.configure(state=tk.DISABLED)

    def clear_transcript(self):
        self.txt.configure(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.configure(state=tk.DISABLED)
        self.partial_var.set("")

    def save_transcript(self):
        content = self.txt.get("1.0", "end-1c")
        if not content.strip():
            self.status_var.set("Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save transcript", defaultextension=".txt",
            initialfile=f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")], parent=self.root)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            self.status_var.set(f"Transcript saved to {path}")
        except OSError as e:
            messagebox.showerror("Save failed", str(e), parent=self.root)

    # ---------- start / stop ----------
    def start_transcription(self):
        if self.session is not None and self.session.is_running():
            return
        name = self.device_var.get()
        desc = self.devices.get(name)
        if desc is None:
            messagebox.showerror("No audio source", "Select a valid audio source first.", parent=self.root)
            return
        sources = [desc]
        mix_name = self.mix_var.get()
        if mix_name in self.devices and self.devices[mix_name] != desc:
            sources.append(self.devices[mix_name])
        engine = config.get("Engine", "type", fallback="vosk")
        err = self._validate_engine(engine)
        if err:
            messagebox.showerror("Cannot start", err, parent=self.root)
            return
        self._set_running_ui(True)
        self.partial_var.set("")
        self.status_var.set("Starting...")
        combo = " + ".join(s[2] or "Mic+System" for s in sources)
        self.append_info(f"--- started ({ENGINE_LABELS.get(engine, engine)} | {combo}) ---")
        self.session = TranscriptionSession(gui_queue, sources)
        self.session.start()

    def _validate_engine(self, engine):
        spec = ENGINES.get(engine)
        if spec is None:
            return f"Unknown engine '{engine}'."
        if not spec["available"]():
            return f"{spec['label']} is not installed.\n\n{spec['install']}"
        return spec["validate"]()

    def stop_transcription(self):
        if self.session is None or not self.session.is_running():
            return
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopping...")
        self.session.stop()

    def _set_running_ui(self, running):
        idle_state = tk.DISABLED if running else tk.NORMAL
        self.start_btn.config(state=tk.DISABLED if (running or not self.devices) else tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL if running else tk.DISABLED)
        self.device_combo.config(state=tk.DISABLED if running else ("readonly" if self.devices else tk.DISABLED))
        self.mix_combo.config(state=tk.DISABLED if running else ("readonly" if self.devices else tk.DISABLED))
        self.engine_combo.config(state=tk.DISABLED if running else ("readonly" if self._engine_display_to_key else tk.DISABLED))
        self.settings_btn.config(state=idle_state)
        self.refresh_btn.config(state=idle_state)

    # ---------- queue pump ----------
    def pump_queue(self):
        try:
            while True:
                kind, payload = gui_queue.get_nowait()
                if kind == "final":
                    ts, text, speaker = payload
                    self.partial_var.set("")
                    self.append_final(ts, text, speaker)
                elif kind == "partial":
                    self.partial_var.set(("… " + payload) if payload else "")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "error":
                    self.status_var.set(payload)
                    self.append_info(f"⚠ {payload}", "warn")
                elif kind == "fatal":
                    self.partial_var.set("")
                    if self.session is not None:
                        self.session.stop()
                    self.status_var.set("Error - stopped.")
                    self.append_info(f"⚠ {payload}", "warn")
                    messagebox.showerror("Transcription error", payload, parent=self.root)
                elif kind == "started":
                    pass  # UI already switched at click time
                elif kind == "stopped":
                    self.partial_var.set("")
                    self._set_running_ui(False)
                    if not self.status_var.get().startswith("Error"):
                        self.status_var.set("Stopped. Idle.")
                    self.append_info("--- stopped ---")
                elif kind == "download_done":
                    ok, label, err = payload
                    parent = self.settings_dialog if self._settings_open() else self.root
                    if ok:
                        self.status_var.set(f"{label.capitalize()} downloaded and ready.")
                        messagebox.showinfo("Download complete",
                                            f"{label.capitalize()} is ready to use.", parent=parent)
                    else:
                        self.status_var.set(f"{label.capitalize()} download failed: {err}")
                        messagebox.showerror("Download failed", err or "Unknown error", parent=parent)
                    if self._settings_open():
                        self.settings_dialog.on_download_finished()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"GUI queue error: {e}")
        finally:
            self.root.after(80, self.pump_queue)

    def _settings_open(self):
        return self.settings_dialog is not None and self.settings_dialog.winfo_exists()

    # ---------- settings ----------
    def open_settings(self):
        if self._settings_open():
            self.settings_dialog.lift()
            return
        self.settings_dialog = SettingsDialog(self)

    def start_vosk_download(self, key):
        threading.Thread(target=download_vosk_model, args=(key, gui_queue),
                         name="VoskDownload", daemon=True).start()

    def start_sherpa_download(self, key):
        threading.Thread(target=download_sherpa_model, args=(key, gui_queue),
                         name="SherpaDownload", daemon=True).start()

    def start_speaker_download(self):
        threading.Thread(target=download_speaker_model, args=(gui_queue,),
                         name="SpeakerDownload", daemon=True).start()

    # ---------- closing ----------
    def on_closing(self):
        if self._settings_open():
            self.settings_dialog.destroy()
        save_config()
        if self.session is not None and self.session.is_running():
            self.session.stop()
            self.root.after(600, self.root.destroy)
        else:
            self.root.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        p = app.palette
        self.title("Settings")
        self.configure(bg=p["bg"])
        self.minsize(620, 540)
        self.transient(app.root)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        # ---- variables ----
        self.engine_var = tk.StringVar(value=config.get("Engine", "type", fallback="vosk"))
        self.vosk_model_var = tk.StringVar(value=config.get("Models", "preferred_vosk_model_type", fallback=DEFAULT_VOSK_MODEL_TYPE))
        self.custom_path_var = tk.StringVar(value=config.get("Paths", "custom_model_path", fallback=""))
        self.backend_var = tk.StringVar(value=config.get("Whisper", "backend", fallback="faster-whisper"))
        self.size_var = tk.StringVar(value=config.get("Whisper", "model_size", fallback=DEFAULT_WHISPER_SIZE))
        self.device_var = tk.StringVar(value=config.get("Whisper", "device", fallback="auto"))
        self.compute_var = tk.StringVar(value=config.get("Whisper", "compute_type", fallback="auto"))
        self.language_var = tk.StringVar(value=config.get("Whisper", "language", fallback="auto"))
        self.vad_var = tk.BooleanVar(value=config.getboolean("Whisper", "vad_filter", fallback=True))
        self.creds_var = tk.StringVar(value=config.get("Engine", "google_cloud_credentials_json", fallback=""))
        self.sherpa_model_var = tk.StringVar(value=config.get("Sherpa", "model", fallback=DEFAULT_SHERPA_MODEL))
        self.sherpa_custom_var = tk.StringVar(value=config.get("Sherpa", "custom_model_dir", fallback=""))
        self.moonshine_var = tk.StringVar(value=config.get("Moonshine", "model", fallback=DEFAULT_MOONSHINE_MODEL))
        self.gweb_lang_var = tk.StringVar(value=config.get("GoogleWeb", "language", fallback="en-US"))
        self.spk_enabled_var = tk.BooleanVar(value=config.getboolean("Speakers", "enabled", fallback=True))
        self.spk_threshold_var = tk.StringVar(value=config.get("Speakers", "similarity_threshold", fallback="0.45"))
        self.spk_max_var = tk.StringVar(value=config.get("Speakers", "max_speakers", fallback="8"))
        self.pause_var = tk.StringVar(value=config.get("Audio", "pause_threshold", fallback="0.7"))
        self.maxphrase_var = tk.StringVar(value=config.get("Audio", "max_phrase_sec", fallback="12.0"))
        self.energy_var = tk.StringVar(value=config.get("Audio", "energy_threshold", fallback="auto"))
        self.logging_var = tk.BooleanVar(value=config.getboolean("Settings", "enable_logging", fallback=True))
        self.logpath_var = tk.StringVar(value=config.get("Audio", "log_file", fallback="live_transcription.log"))
        self.fontsize_var = tk.StringVar(value=config.get("Settings", "font_size", fallback="11"))
        self.overlay_opacity_var = tk.DoubleVar(value=config.getfloat("Settings", "overlay_opacity", fallback=85))

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

        nb.add(self._build_engine_tab(nb), text="Engine")
        nb.add(self._build_streaming_tab(nb), text="Streaming")
        nb.add(self._build_whisper_tab(nb), text="Whisper")
        nb.add(self._build_online_tab(nb), text="Online")
        nb.add(self._build_speakers_tab(nb), text="Speakers")
        nb.add(self._build_audio_tab(nb), text="Audio")
        nb.add(self._build_appearance_tab(nb), text="Appearance")

        btns = ttk.Frame(self, padding=10)
        btns.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(btns, text="Save", style="Accent.TButton", command=self.save).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

        set_titlebar_dark(self, config.get("Settings", "theme", fallback="dark") == "dark")
        self.update_idletasks()

    # -- tabs --
    def _build_engine_tab(self, parent):
        tab = ttk.Frame(parent, padding=14)
        for key, spec in ENGINES.items():
            rb = ttk.Radiobutton(tab, text=spec["title"], value=key, variable=self.engine_var)
            rb.pack(anchor="w", pady=(6, 0))
            desc = spec["desc"]
            if not spec["available"]():
                rb.config(state=tk.DISABLED)
                desc = f"Not installed - {spec['install']}"
            ttk.Label(tab, text="     " + desc, style="Muted.TLabel", wraplength=520,
                      justify=tk.LEFT).pack(anchor="w")
        return tab

    def _build_streaming_tab(self, parent):
        tab = ttk.Frame(parent, padding=10)

        vosk_frame = ttk.Labelframe(tab, text="Vosk models", padding=8)
        vosk_frame.pack(fill=tk.X, pady=(0, 8))
        self._vosk_radios = {}
        for key, info in MODEL_INFO.items():
            rb = ttk.Radiobutton(vosk_frame, value=key, variable=self.vosk_model_var)
            rb.pack(anchor="w", padx=4, pady=1)
            self._vosk_radios[key] = rb
        self._refresh_vosk_labels()
        ttk.Radiobutton(vosk_frame, text="Custom model folder:", value="custom",
                        variable=self.vosk_model_var).pack(anchor="w", padx=4, pady=(4, 1))
        row = ttk.Frame(vosk_frame)
        row.pack(fill=tk.X, padx=4)
        ttk.Entry(row, textvariable=self.custom_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_custom).pack(side=tk.LEFT, padx=(6, 0))
        self.vosk_download_btn = ttk.Button(vosk_frame, text="Download selected Vosk model",
                                            command=self._download_selected)
        self.vosk_download_btn.pack(anchor="w", padx=4, pady=(6, 2))
        ttk.Label(vosk_frame, text="More languages: alphacephei.com/vosk/models",
                  style="Muted.TLabel").pack(anchor="w", padx=4)

        sherpa_frame = ttk.Labelframe(tab, text="Sherpa streaming models (sherpa-onnx)", padding=8)
        sherpa_frame.pack(fill=tk.X)
        self._sherpa_radios = {}
        for key, info in SHERPA_MODELS.items():
            rb = ttk.Radiobutton(sherpa_frame, value=key, variable=self.sherpa_model_var)
            rb.pack(anchor="w", padx=4, pady=1)
            self._sherpa_radios[key] = rb
        self._refresh_sherpa_labels()
        ttk.Radiobutton(sherpa_frame, text="Custom model folder:", value="custom",
                        variable=self.sherpa_model_var).pack(anchor="w", padx=4, pady=(4, 1))
        row2 = ttk.Frame(sherpa_frame)
        row2.pack(fill=tk.X, padx=4)
        ttk.Entry(row2, textvariable=self.sherpa_custom_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row2, text="Browse...", command=self._browse_sherpa_custom).pack(side=tk.LEFT, padx=(6, 0))
        self.sherpa_download_btn = ttk.Button(sherpa_frame, text="Download selected Sherpa model",
                                              command=self._download_sherpa_selected)
        self.sherpa_download_btn.pack(anchor="w", padx=4, pady=(6, 2))
        if not HAVE_SHERPA:
            self.sherpa_download_btn.config(state=tk.DISABLED)
            ttk.Label(sherpa_frame, text="sherpa-onnx is not installed (pip install sherpa-onnx).",
                      style="Muted.TLabel").pack(anchor="w", padx=4)

        if not HAVE_REQUESTS:
            for btn in (self.vosk_download_btn, self.sherpa_download_btn):
                btn.config(state=tk.DISABLED)
            ttk.Label(tab, text="Downloads need the 'requests' package (pip install requests).",
                      style="Muted.TLabel").pack(anchor="w")
        return tab

    def _build_whisper_tab(self, parent):
        tab = ttk.Frame(parent, padding=14)
        tab.columnconfigure(1, weight=1)

        backends = []
        if HAVE_FASTER_WHISPER:
            backends.append("faster-whisper")
        if HAVE_OPENAI_WHISPER:
            backends.append("openai-whisper")
        ttk.Label(tab, text="Backend:").grid(row=0, column=0, sticky="w", pady=4)
        self.backend_combo = ttk.Combobox(tab, textvariable=self.backend_var,
                                          values=backends or ["faster-whisper"],
                                          state="readonly" if backends else tk.DISABLED, width=20)
        self.backend_combo.grid(row=0, column=1, sticky="w", pady=4)
        self.backend_combo.bind("<<ComboboxSelected>>", self._on_backend_change)

        ttk.Label(tab, text="Model:").grid(row=1, column=0, sticky="w", pady=4)
        self.size_combo = ttk.Combobox(tab, textvariable=self.size_var, width=34)
        self.size_combo.grid(row=1, column=1, sticky="w", pady=4)
        self._on_backend_change()

        ttk.Label(tab, text="Device:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(tab, textvariable=self.device_var, values=WHISPER_DEVICES,
                     state="readonly", width=20).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(tab, text="Compute type:").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Combobox(tab, textvariable=self.compute_var, values=WHISPER_COMPUTE_TYPES,
                     state="readonly", width=20).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="(faster-whisper only; 'int8' is a good CPU default, 'float16' for GPU)",
                  style="Muted.TLabel").grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Label(tab, text="Language:").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.language_var, width=22).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="('auto' to detect, or a code like en, de, fr, es...)",
                  style="Muted.TLabel").grid(row=6, column=0, columnspan=2, sticky="w")

        ttk.Checkbutton(tab, text="Use built-in VAD filter (faster-whisper, recommended)",
                        variable=self.vad_var).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(tab, text="Tips: 'large-v3-turbo' and 'distil-large-v3' are the newest free models -\n"
                            "near large-v3 accuracy at several times the speed. Models download on first use.\n"
                            "Future models: with faster-whisper you can type ANY Hugging Face CTranslate2\n"
                            "repo id in the Model box (e.g. user/some-new-whisper-ct2).",
                  style="Muted.TLabel", justify=tk.LEFT).grid(row=8, column=0, columnspan=2, sticky="w", pady=(12, 0))

        ttk.Separator(tab).grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(tab, text="Moonshine model:").grid(row=10, column=0, sticky="w", pady=4)
        ttk.Combobox(tab, textvariable=self.moonshine_var, values=MOONSHINE_MODELS,
                     state="readonly" if HAVE_MOONSHINE else tk.DISABLED,
                     width=20).grid(row=10, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="(Moonshine engine - English only; 'base' is more accurate, 'tiny' is fastest.)",
                  style="Muted.TLabel").grid(row=11, column=0, columnspan=2, sticky="w")
        return tab

    def _build_online_tab(self, parent):
        tab = ttk.Frame(parent, padding=14)

        web_frame = ttk.Labelframe(tab, text="Google Web Speech (free)", padding=8)
        web_frame.pack(fill=tk.X, pady=(0, 10))
        row0 = ttk.Frame(web_frame)
        row0.pack(fill=tk.X)
        ttk.Label(row0, text="Language:").pack(side=tk.LEFT)
        ttk.Entry(row0, textvariable=self.gweb_lang_var, width=12).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(web_frame, text="A BCP-47 code such as en-US, en-GB, de-DE, fr-FR...\n"
                                  "No key or account needed; audio is sent to Google.",
                  style="Muted.TLabel", justify=tk.LEFT).pack(anchor="w", pady=(6, 0))

        cloud_frame = ttk.Labelframe(tab, text="Google Cloud Speech-to-Text", padding=8)
        cloud_frame.pack(fill=tk.X)
        ttk.Label(cloud_frame, text="Credentials JSON file:").pack(anchor="w")
        row = ttk.Frame(cloud_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Entry(row, textvariable=self.creds_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_creds).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(cloud_frame, text="Create a service account with Speech-to-Text access in Google Cloud Console\n"
                                    "and download its JSON key. Note: this engine sends audio to Google.",
                  style="Muted.TLabel", justify=tk.LEFT).pack(anchor="w", pady=(6, 0))
        return tab

    def _build_speakers_tab(self, parent):
        tab = ttk.Frame(parent, padding=14)
        tab.columnconfigure(1, weight=1)
        ttk.Checkbutton(tab, text="Label speakers automatically (Speaker 1, Speaker 2, ...)",
                        variable=self.spk_enabled_var).grid(row=0, column=0, columnspan=2, sticky="w")

        self.spk_status_label = ttk.Label(tab, style="Muted.TLabel")
        self.spk_status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.spk_download_btn = ttk.Button(tab, text="Download speaker model (~28 MB)",
                                           command=self._download_speaker_model)
        self.spk_download_btn.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 10))
        if not (HAVE_SHERPA and HAVE_REQUESTS):
            self.spk_download_btn.config(state=tk.DISABLED)
        self._refresh_speaker_status()

        ttk.Label(tab, text="Similarity threshold:").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=0.20, to=0.80, increment=0.05, textvariable=self.spk_threshold_var,
                    width=8).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="Lower it if one person gets split into several speakers;\n"
                            "raise it if different people get merged into one.",
                  style="Muted.TLabel", justify=tk.LEFT).grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Label(tab, text="Max speakers:").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=2, to=16, increment=1, textvariable=self.spk_max_var,
                    width=8).grid(row=5, column=1, sticky="w", pady=4)

        ttk.Button(tab, text="Reset learned speakers",
                   command=self._reset_speakers).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(tab, text="Works with every engine. Voices are clustered as they speak - numbering\n"
                            "starts fresh after a reset or app restart. Needs sherpa-onnx installed.",
                  style="Muted.TLabel", justify=tk.LEFT).grid(row=7, column=0, columnspan=2, sticky="w", pady=(12, 0))
        return tab

    def _build_audio_tab(self, parent):
        tab = ttk.Frame(parent, padding=14)
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="Pause threshold (s):").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=0.3, to=3.0, increment=0.1, textvariable=self.pause_var,
                    width=8).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="Silence needed before a phrase is sent for transcription (Whisper/Google).",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w")

        ttk.Label(tab, text="Max phrase length (s):").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=4, to=30, increment=1, textvariable=self.maxphrase_var,
                    width=8).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(tab, text="Energy threshold:").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.energy_var, width=10).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="'auto' calibrates from the first moment of audio, or set a fixed RMS value\n"
                            "(e.g. 0.01). Raise it if background noise triggers false phrases.",
                  style="Muted.TLabel", justify=tk.LEFT).grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Separator(tab).grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Checkbutton(tab, text="Log transcripts to file", variable=self.logging_var).grid(
            row=6, column=0, columnspan=2, sticky="w")
        row7 = ttk.Frame(tab)
        row7.grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Entry(row7, textvariable=self.logpath_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row7, text="Browse...", command=self._browse_log).pack(side=tk.LEFT, padx=(6, 0))
        return tab

    def _build_appearance_tab(self, parent):
        tab = ttk.Frame(parent, padding=14)
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="Transcript font size:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=8, to=24, increment=1, textvariable=self.fontsize_var,
                    width=6).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="Overlay mode opacity (%):").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Scale(tab, from_=30, to=100, variable=self.overlay_opacity_var,
                  length=180).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="Theme and window opacity are on the main toolbar.\n"
                            "Overlay mode hides the controls, pins the window on top and applies\n"
                            "the opacity above - press Esc to restore.",
                  style="Muted.TLabel", justify=tk.LEFT).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        return tab

    # -- tab callbacks --
    def _on_backend_change(self, event=None):
        fw = self.backend_var.get() == "faster-whisper"
        sizes = FASTER_WHISPER_SIZES if fw else OPENAI_WHISPER_SIZES
        # faster-whisper accepts any Hugging Face CTranslate2 repo id, so the
        # box stays editable there; openai-whisper only knows fixed names.
        self.size_combo.config(values=sizes, state="normal" if fw else "readonly")
        if not fw and self.size_var.get() not in sizes:
            self.size_var.set(DEFAULT_WHISPER_SIZE)

    def _refresh_vosk_labels(self):
        for key, rb in self._vosk_radios.items():
            mark = "  ✓ downloaded" if vosk_model_downloaded(key) else ""
            rb.config(text=f"{MODEL_INFO[key]['description']}{mark}")

    def _download_selected(self):
        key = self.vosk_model_var.get()
        if key not in MODEL_INFO:
            messagebox.showwarning("Pick a model", "Select one of the standard models to download.", parent=self)
            return
        if vosk_model_downloaded(key):
            messagebox.showinfo("Already downloaded", f"The '{key}' model is already downloaded.", parent=self)
            return
        self.vosk_download_btn.config(state=tk.DISABLED, text="Downloading...")
        self.app.start_vosk_download(key)

    def _refresh_sherpa_labels(self):
        for key, rb in self._sherpa_radios.items():
            mark = "  ✓ downloaded" if sherpa_model_downloaded(key) else ""
            rb.config(text=f"{SHERPA_MODELS[key]['description']}{mark}")

    def _download_sherpa_selected(self):
        key = self.sherpa_model_var.get()
        if key not in SHERPA_MODELS:
            messagebox.showwarning("Pick a model", "Select one of the standard models to download.", parent=self)
            return
        if sherpa_model_downloaded(key):
            messagebox.showinfo("Already downloaded", f"The '{key}' model is already downloaded.", parent=self)
            return
        self.sherpa_download_btn.config(state=tk.DISABLED, text="Downloading...")
        self.app.start_sherpa_download(key)

    def _browse_sherpa_custom(self):
        d = filedialog.askdirectory(title="Select sherpa-onnx model folder", parent=self)
        if d:
            self.sherpa_custom_var.set(d)
            self.sherpa_model_var.set("custom")

    def _refresh_speaker_status(self):
        if not HAVE_SHERPA:
            text = "Needs sherpa-onnx installed (pip install sherpa-onnx)."
        elif os.path.exists(speaker_model_path()):
            text = "Speaker model: ✓ downloaded and ready."
            if self.spk_download_btn.winfo_exists():
                self.spk_download_btn.config(state=tk.DISABLED)
        else:
            text = "Speaker model: not downloaded yet - labels stay off until it is."
        self.spk_status_label.config(text=text)

    def _download_speaker_model(self):
        if os.path.exists(speaker_model_path()):
            self._refresh_speaker_status()
            return
        self.spk_download_btn.config(state=tk.DISABLED, text="Downloading...")
        self.app.start_speaker_download()

    def _reset_speakers(self):
        speaker_registry.reset()
        self.app.status_var.set("Learned speakers reset - numbering starts fresh.")

    def on_download_finished(self):
        if self.vosk_download_btn.winfo_exists():
            self.vosk_download_btn.config(state=tk.NORMAL, text="Download selected Vosk model")
        if HAVE_SHERPA and HAVE_REQUESTS and self.sherpa_download_btn.winfo_exists():
            self.sherpa_download_btn.config(state=tk.NORMAL, text="Download selected Sherpa model")
        if self.spk_download_btn.winfo_exists():
            self.spk_download_btn.config(text="Download speaker model (~28 MB)")
            if HAVE_SHERPA and HAVE_REQUESTS and not os.path.exists(speaker_model_path()):
                self.spk_download_btn.config(state=tk.NORMAL)
        self._refresh_vosk_labels()
        self._refresh_sherpa_labels()
        self._refresh_speaker_status()

    def _browse_custom(self):
        d = filedialog.askdirectory(title="Select Vosk model folder", parent=self)
        if d:
            self.custom_path_var.set(d)
            self.vosk_model_var.set("custom")

    def _browse_creds(self):
        f = filedialog.askopenfilename(title="Select Google Cloud credentials JSON",
                                       filetypes=[("JSON files", "*.json"), ("All files", "*.*")], parent=self)
        if f:
            self.creds_var.set(f)

    def _browse_log(self):
        f = filedialog.asksaveasfilename(title="Select log file", defaultextension=".log",
                                         initialfile=os.path.basename(self.logpath_var.get()) or "live_transcription.log",
                                         filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
                                         parent=self)
        if f:
            self.logpath_var.set(f)

    # -- save --
    def save(self):
        engine = self.engine_var.get()
        if engine == "vosk" and self.vosk_model_var.get() == "custom" and not self.custom_path_var.get().strip():
            messagebox.showwarning("Missing path", "Set a custom Vosk model folder or pick a standard model.", parent=self)
            return
        if engine == "sherpa" and self.sherpa_model_var.get() == "custom" and not self.sherpa_custom_var.get().strip():
            messagebox.showwarning("Missing path", "Set a custom Sherpa model folder or pick a standard model.", parent=self)
            return
        if engine == "google_cloud":
            creds = self.creds_var.get().strip()
            if not creds:
                messagebox.showwarning("Missing credentials", "Google Cloud needs a credentials JSON file.", parent=self)
                return
            check = creds if os.path.isabs(creds) else os.path.join(get_base_path(), creds)
            if not os.path.exists(check):
                messagebox.showwarning("File not found", f"Credentials file not found:\n{check}", parent=self)
                return
        try:
            pause = float(self.pause_var.get())
            maxphrase = float(self.maxphrase_var.get())
            fontsize = int(float(self.fontsize_var.get()))
        except ValueError:
            messagebox.showwarning("Invalid value", "Pause threshold, max phrase length and font size must be numbers.", parent=self)
            return
        energy = self.energy_var.get().strip().lower()
        if energy not in ("", "auto"):
            try:
                float(energy)
            except ValueError:
                messagebox.showwarning("Invalid value", "Energy threshold must be 'auto' or a number (e.g. 0.01).", parent=self)
                return
        try:
            spk_threshold = float(self.spk_threshold_var.get())
            spk_max = int(float(self.spk_max_var.get()))
        except ValueError:
            messagebox.showwarning("Invalid value", "Speaker threshold and max speakers must be numbers.", parent=self)
            return

        config.set("Engine", "type", engine)
        config.set("Engine", "google_cloud_credentials_json", self.creds_var.get().strip())
        config.set("Models", "preferred_vosk_model_type", self.vosk_model_var.get())
        config.set("Paths", "custom_model_path", self.custom_path_var.get().strip())
        config.set("Whisper", "backend", self.backend_var.get())
        config.set("Whisper", "model_size", self.size_var.get().strip() or DEFAULT_WHISPER_SIZE)
        config.set("Whisper", "device", self.device_var.get())
        config.set("Whisper", "compute_type", self.compute_var.get())
        config.set("Whisper", "language", self.language_var.get().strip() or "auto")
        config.set("Whisper", "vad_filter", str(bool(self.vad_var.get())))
        config.set("Sherpa", "model", self.sherpa_model_var.get())
        config.set("Sherpa", "custom_model_dir", self.sherpa_custom_var.get().strip())
        config.set("Moonshine", "model", self.moonshine_var.get())
        config.set("GoogleWeb", "language", self.gweb_lang_var.get().strip() or "en-US")
        config.set("Speakers", "enabled", str(bool(self.spk_enabled_var.get())))
        config.set("Speakers", "similarity_threshold", f"{max(0.2, min(0.9, spk_threshold)):.2f}")
        config.set("Speakers", "max_speakers", str(max(2, min(16, spk_max))))
        config.set("Audio", "pause_threshold", f"{max(0.2, min(5.0, pause)):.2f}")
        config.set("Audio", "max_phrase_sec", f"{max(3.0, min(60.0, maxphrase)):.1f}")
        config.set("Audio", "energy_threshold", energy or "auto")
        config.set("Settings", "enable_logging", str(bool(self.logging_var.get())))
        config.set("Audio", "log_file", self.logpath_var.get().strip() or "live_transcription.log")
        config.set("Settings", "overlay_opacity", f"{self.overlay_opacity_var.get():.0f}")

        save_config()
        self.app.set_font_size(max(8, min(24, fontsize)))
        if engine in ENGINE_LABELS and engine in available_engines():
            self.app.engine_var.set(ENGINE_LABELS[engine])
        self.app.status_var.set("Settings saved.")
        self.destroy()


# --- Main ------------------------------------------------------------------------
def main():
    load_config()
    # Goes to app_errors.log in frozen builds - first thing to check when
    # someone reports a missing engine.
    print(f"{APP_TITLE} starting ({datetime.now():%Y-%m-%d %H:%M:%S}) - "
          f"engines available: {', '.join(available_engines()) or 'NONE'}")
    root = tk.Tk()
    app = TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
