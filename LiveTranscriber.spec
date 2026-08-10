# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Live Transcriber.
# Build with:  pyinstaller LiveTranscriber.spec
# Models are NOT bundled - they download next to the exe on first use,
# which keeps the zip small enough to share.

from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_dynamic_libs)

datas, binaries, hiddenimports = [], [], []

# Packages whose DLLs / data files live inside the package folder.
for pkg in ("vosk", "sherpa_onnx"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += collect_data_files("faster_whisper")     # silero VAD assets
datas += collect_data_files("speech_recognition")  # flac binaries
datas += collect_data_files("moonshine_onnx")      # tokenizer assets
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")

# moonshine_onnx is imported lazily at runtime, so name it explicitly.
hiddenimports += ["moonshine_onnx"]

a = Analysis(
    ["Transcriber_GUI.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Lazy imports we never hit at runtime (librosa path-loading in
    # moonshine) drag in numba/scipy/sklearn - keep them out.
    excludes=["librosa", "numba", "scipy", "sklearn", "matplotlib",
              "torch", "IPython", "pytest", "soundfile", "audioread",
              "pooch", "msgpack"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiveTranscriber",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LiveTranscriber",
)
