# PyInstaller build for the macOS app. Driven by ./build_macos.sh.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# mlx ships compiled Metal libraries, and mlx_whisper carries the mel filter
# bank and tokenizer files as package data; neither is found by static
# analysis, so both are collected wholesale.
# webview carries the Cocoa backend and its JS bridge as package data.
# speechbrain/torch are deliberately NOT bundled. They carry the ECAPA
# speaker embeddings that make mono diarization good, but PyInstaller cannot
# scan speechbrain's module tree without crashing, and they would add some
# 350 MB. The app therefore uses the MFCC fingerprint on mono files and says
# so on a !! line; run from source with speechbrain installed to get ECAPA.
for pkg in ("mlx", "mlx_whisper", "numba", "llvmlite", "tiktoken", "soundfile",
            "webview"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# The stages are launched through runpy, so nothing imports them by name.
hiddenimports += [
    # pulling multiprocessing in by name makes PyInstaller apply its
    # runtime hook, which is what teaches the frozen binary to recognise a
    # re-exec as a worker rather than a fresh launch
    "multiprocessing", "multiprocessing.resource_tracker",
    # the WebView window and the Cocoa bridge it sits on
    "webview", "webview.platforms.cocoa", "objc", "Foundation", "AppKit",
    "WebKit", "Quartz",
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
    # torch is a declared dependency of mlx-whisper but only its checkpoint
    # converter uses it. Dropping it takes 339 MB off the bundle.
    excludes=["torch", "torchaudio", "torchvision", "speechbrain",
              "matplotlib", "tkinter",
              "PIL", "pandas", "IPython", "pytest", "transformers",
              # the other backend; this build uses mlx
              "faster_whisper", "ctranslate2"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AudioTranscribe",
          console=False, target_arch="arm64")

coll = COLLECT(exe, a.binaries, a.datas, name="AudioTranscribe")

app = BUNDLE(
    coll,
    name="AudioTranscribe.app",
    bundle_identifier="local.audio.transcribe",
    info_plist={
        "CFBundleName": "AudioTranscribe",
        "CFBundleDisplayName": "Audio Transcription & Annotation",
        "CFBundleShortVersionString": "1.0.0",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        # it is a local server with a browser front end, not a background agent
        "LSBackgroundOnly": False,
    },
)
