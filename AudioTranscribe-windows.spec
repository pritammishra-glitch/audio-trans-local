# PyInstaller build for Windows. Driven by build_windows.bat.
#
# The macOS spec collects mlx; this one collects faster-whisper and
# ctranslate2 instead. Everything else about the app is identical, because
# the backend choice lives behind src/asr_backend.py.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# ctranslate2 ships compiled DLLs, faster-whisper carries the Silero VAD
# model and the tokenizer assets; static analysis finds none of it.
for pkg in ("faster_whisper", "ctranslate2", "tokenizers", "onnxruntime",
            "soundfile", "av", "webview"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass   # av and onnxruntime are optional extras

# The stages are launched through runpy, so nothing imports them by name.
hiddenimports += [
    "multiprocessing", "multiprocessing.resource_tracker",
    "webview", "webview.platforms.edgechromium", "clr_loader", "pythonnet",
    "pipeline", "webapp", "entities", "precheck", "asr_backend",
    "stage0_prep", "stage1_asr", "stage2_speakers", "stage2_mono",
    "stage3_acoustics", "stage4_events", "stage5_speaker_meta",
    "stage6_annotate", "stage7_merge", "stage7b_language", "stage8_format",
]

a = Analysis(
    ["src/app.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["mlx", "mlx_whisper", "torch", "torchaudio", "torchvision",
              "matplotlib", "tkinter", "PIL", "pandas", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

# console=False: the window is the app, as on macOS
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="AudioTranscribe", console=False)

coll = COLLECT(exe, a.binaries, a.datas, name="AudioTranscribe")
