"""Run the stages, in or out of a PyInstaller bundle.

run.sh is the shell version of this and stays the way to drive the pipeline
from a terminal. A packaged .app has no shell script and no .venv, so the same
sequence lives here in Python.

The one thing worth preserving is why each stage is a separate process: a stage
exits before the next one loads, so macOS reclaims its memory and two large
models are never resident at once. Frozen, the binary re-invokes itself with
`--stage <name>`; unfrozen, it runs the stage file with the current
interpreter. Either way it is still a fresh process per stage.
"""
import os, pathlib, subprocess, sys, threading

SRC = pathlib.Path(__file__).resolve().parent
FROZEN = getattr(sys, "frozen", False)

# (label, stage module, extra args) - stage 7b needs the second language, and
# stage 2 falls back to clustering when the file is mono.
STAGES = [
    ("prepare audio",      "stage0_prep",        "prep"),
    ("transcription",      "stage1_asr",         "lang"),
    ("speaker separation", "stage2_speakers",    "mono_fallback"),
    ("pauses and silences", "stage3_acoustics",  ""),
    ("environment events", "stage4_events",      ""),
    ("speaker metadata",   "stage5_speaker_meta", ""),
    ("merge into turns",   "stage7_merge",       ""),
    ("language per turn",  "stage7b_language",   "second"),
    ("annotation",         "stage6_annotate",    "fast"),
    ("format",             "stage8_format",      ""),
]


# Stage processes in flight, so quitting the app can take them with it
# instead of leaving whisper running headless after the window is gone.
_children = set()
_children_lock = threading.Lock()


def kill_children():
    with _children_lock:
        procs = list(_children)
        _children.clear()
    for pr in procs:
        try:
            pr.terminate()
        except Exception:
            pass


def stage_cmd(module, *args):
    if FROZEN:
        return [sys.executable, "--stage", module, *args]
    return [sys.executable, str(SRC / f"{module}.py"), *args]


def data_root():
    """Where audio and results live.

    A bundle may sit somewhere read-only - /Applications, Program Files - so
    a frozen app writes to the platform's per-user data folder rather than
    next to the executable. Run from source, everything stays in the project
    folder where it has always been.
    """
    if not FROZEN:
        d = SRC.parent
    elif sys.platform == "darwin":
        d = pathlib.Path.home() / "Library/Application Support/AudioTranscribe"
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (pathlib.Path.home() / "AppData/Local")
        d = pathlib.Path(base) / "AudioTranscribe"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (pathlib.Path.home() / ".local/share")
        d = pathlib.Path(base) / "AudioTranscribe"
    (d / "audio").mkdir(parents=True, exist_ok=True)
    (d / "out").mkdir(parents=True, exist_ok=True)
    return d


class _Done:
    """Just enough of a CompletedProcess for the caller."""
    def __init__(self, returncode):
        self.returncode = returncode


def _spawn(cmd, env, log):
    """Run one stage, streaming its output, and keep it killable."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with _children_lock:
        _children.add(proc)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line and "Fetching" not in line and "warn" not in line.lower():
                log(line)
        proc.wait()
    finally:
        with _children_lock:
            _children.discard(proc)
    return _Done(proc.returncode)


def run(audio_path, workdir, language="auto", second="", fast=False, log=print):
    """Run every stage in order. Yields nothing; reports through `log`."""
    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OLLAMA_KEEP_ALIVE="0")

    for label, module, kind in STAGES:
        if kind == "prep":
            args = [str(audio_path), str(workdir)]
        elif kind == "lang":
            args = [str(workdir), language or "auto"]
        elif kind == "second":
            args = [str(workdir), second or ""]
        elif kind == "fast":
            args = [str(workdir)] + (["--fast"] if fast else [])
        else:
            args = [str(workdir)]

        log(f"== {label}")
        proc = _spawn(stage_cmd(module, *args), env, log)

        if proc.returncode != 0:
            if kind == "mono_fallback":
                # stereo split unavailable: this is a mono recording
                log("== speaker separation (mono fallback)")
                alt = _spawn(stage_cmd("stage2_mono", str(workdir)), env, log)
                if alt.returncode != 0:
                    raise RuntimeError(f"{label} failed")
                continue
            if module == "stage6_annotate" and proc.returncode == 3:
                log("!! halted: a card number with CVV was spoken")
                continue
            raise RuntimeError(f"{label} failed (exit {proc.returncode})")

    return workdir / "transcript.txt"
