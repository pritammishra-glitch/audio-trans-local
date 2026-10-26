"""Stage 0 - prepare the two working copies of the audio.

  original.wav  the input untouched in channel count and rate; stages 2-5 need
                the separate channels
  mono16k.wav   16 kHz mono, which is what whisper expects

The downmix is done here rather than with `afconvert -c 1`, which discards a
channel instead of summing it - on a stereo call that silently loses one
speaker entirely.
"""
import shutil, subprocess, sys, pathlib, tempfile
import numpy as np, soundfile as sf
from scipy import signal

TARGET_SR = 16000


def decoders():
    """External decoders to try, best first, for what soundfile cannot read.

    ffmpeg is the portable one and is what Windows and Linux use. afconvert
    is CoreAudio's and ships with macOS, so a Mac needs nothing installed.
    """
    tools = []
    if shutil.which("ffmpeg"):
        tools.append(lambda src, dst: ["ffmpeg", "-nostdin", "-y", "-i",
                                       str(src), "-c:a", "pcm_s16le", str(dst)])
    if sys.platform == "darwin" and shutil.which("afconvert"):
        tools.append(lambda src, dst: ["afconvert", "-f", "WAVE", "-d",
                                       "LEI16", str(src), str(dst)])
    return tools


def load(path: pathlib.Path):
    try:
        return sf.read(path, always_2d=True, dtype="float32")
    except Exception as first:
        # Compressed formats soundfile was not built for: decode to wav first.
        tmp = pathlib.Path(tempfile.mkdtemp()) / "decoded.wav"
        for build in decoders():
            try:
                subprocess.run(build(path, tmp), check=True,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return sf.read(tmp, always_2d=True, dtype="float32")
            except Exception:
                continue
        raise RuntimeError(
            f"cannot read {path.name}. soundfile said: {first}. Install "
            f"ffmpeg and put it on PATH, or convert the file to .wav."
        ) from first


if __name__ == "__main__":
    src, wd = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    wd.mkdir(parents=True, exist_ok=True)
    audio, sr = load(src)

    sf.write(wd / "original.wav", audio, sr, subtype="PCM_16")

    def to_16k(x):
        if sr != TARGET_SR:
            g = np.gcd(int(sr), TARGET_SR)
            x = signal.resample_poly(x, TARGET_SR // g, int(sr) // g)
        peak = float(np.abs(x).max()) or 1.0
        return (x / peak if peak > 1.0 else x).astype(np.float32)

    sf.write(wd / "mono16k.wav", to_16k(audio.mean(axis=1)), TARGET_SR, subtype="PCM_16")

    # One file per channel as well. Transcribing each speaker's own track beats
    # transcribing the mix: whisper hallucinates when two voices overlap, and
    # the channel then names the speaker exactly instead of by energy.
    extra = ""
    if audio.shape[1] == 2:
        for ch in (0, 1):
            sf.write(wd / f"ch{ch}_16k.wav", to_16k(audio[:, ch]), TARGET_SR,
                     subtype="PCM_16")
        extra = " + ch0/ch1_16k.wav"

    print(f"stage0: {audio.shape[1]} ch @ {sr} Hz, {len(audio)/sr:.1f}s "
          f"-> original.wav + mono16k.wav{extra}")
