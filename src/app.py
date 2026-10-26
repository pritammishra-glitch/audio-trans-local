"""Entry point for the packaged app.

Two jobs in one binary:

  app --stage stage1_asr <args>   run a single pipeline stage
  app                             open the app window

The stage mode is what lets a frozen bundle keep the one-process-per-stage
design without a shell script: the orchestrator re-invokes this same
executable, so each stage still gets a fresh process that exits and frees its
model.

The window is a native WebView, not a browser tab, so the app behaves like an
application: closing the window quits it, and quitting takes any running
transcription with it rather than leaving whisper working in the background.
"""
import multiprocessing
# Must run before anything else. A frozen app re-executes its own binary for
# multiprocessing helpers, and huggingface_hub starts one while downloading
# the model. Without this the child re-enters the app instead of the helper
# and the whole thing hangs.
multiprocessing.freeze_support()

import atexit, os, pathlib, runpy, socket, sys, threading, time, webbrowser

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TITLE = "Audio Transcription & Annotation"

STAGES = {
    "stage0_prep", "stage1_asr", "stage2_speakers", "stage2_mono",
    "stage3_acoustics", "stage4_events", "stage5_speaker_meta",
    "stage6_annotate", "stage7_merge", "stage7b_language", "stage8_format",
}


def run_stage(argv):
    name = argv[0]
    if name not in STAGES:
        sys.exit(f"unknown stage {name!r}")
    sys.argv = [name] + list(argv[1:])
    runpy.run_module(name, run_name="__main__")


def free_port(preferred=5001):
    """Use the usual port when it is free, any free one when it is not.

    Two copies of the app, or anything else already on 5001, should not stop
    it starting.
    """
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


def wait_until_up(url, timeout=30):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def shutdown():
    """Leave nothing running behind the window."""
    try:
        import pipeline
        pipeline.kill_children()
    except Exception:
        pass


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--stage":
        return run_stage(sys.argv[2:])

    from webapp import app

    port = int(os.environ.get("PORT") or free_port())
    url = f"http://127.0.0.1:{port}"
    atexit.register(shutdown)

    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port,
                               debug=False, threaded=True),
        daemon=True).start()
    wait_until_up(url)

    if os.environ.get("NO_WINDOW") == "1":
        print(f"{TITLE}\nopen {url}\n(Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        return shutdown()

    try:
        import webview
    except Exception:
        # No WebView runtime: fall back to the browser rather than fail.
        print(f"{TITLE}\nopen {url}\n(close this window to stop)")
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        return shutdown()

    webview.create_window(TITLE, url, width=1060, height=860,
                          min_size=(820, 600))
    webview.start()        # blocks until the window is closed
    shutdown()             # and then the app is really gone


if __name__ == "__main__":
    main()
