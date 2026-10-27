"""One speech-recognition interface, two implementations.

mlx-whisper runs whisper on the Mac GPU and is the fastest option by a wide
margin, but it only exists for Apple Silicon. faster-whisper runs the same
large-v3-turbo weights through CTranslate2 on any CPU, and on an NVIDIA GPU
if one is present, which is what makes Windows, Linux and Intel Macs possible.

The backend is chosen automatically. Set ASR_BACKEND=mlx or
ASR_BACKEND=faster-whisper to force one, which is also how the faster-whisper
path gets tested on a Mac.

Everything here takes and returns plain numpy: float32 mono at 16 kHz. Neither
backend is handed a file path, so no ffmpeg is needed at this layer.
"""
import os, sys

SR = 16000
MODEL_MLX = "mlx-community/whisper-large-v3-turbo"
MODEL_FW = "large-v3-turbo"

# Whisper's fallback ladder re-decodes a segment at rising temperature when it
# fails the quality thresholds. Which setting is right depends on the audio,
# and both directions were measured on real files:
#
#   Narrowband telephone audio (8 kHz): without the ladder a failed segment is
#   simply dropped, and words go missing.
#   Clean wideband audio: with the ladder, segments that would have decoded
#   fine get re-rolled at temperature and come back worse - "9014" became
#   "90114" and "Brooks" became "Broog" on a file that was previously exact.
#
# So the ladder is used only where it helps. stage 1 decides from the source
# sample rate.
TEMPERATURES_ROBUST = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TEMPERATURES_EXACT = 0.0
COMPRESSION_RATIO_THRESHOLD = 2.4   # above this the text is repetitive
LOGPROB_THRESHOLD = -1.0            # below this the decode is not trusted
NO_SPEECH_THRESHOLD = 0.6

# Domain words whisper mishears on telephone audio. Seeded as a decoding hint.
HOTWORDS = os.environ.get("ASR_HOTWORDS", "")

_model = None       # the loaded backend model, cached per process
_backend = None


def backend():
    """'mlx' or 'faster-whisper'."""
    global _backend
    if _backend:
        return _backend
    want = (os.environ.get("ASR_BACKEND") or "auto").lower()
    if want in ("mlx", "faster-whisper"):
        _backend = want
    else:
        try:
            import mlx.core  # noqa: F401
            _backend = "mlx"
        except Exception:
            _backend = "faster-whisper"
    return _backend


def _fw_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        device = os.environ.get("ASR_DEVICE", "auto")
        compute = os.environ.get("ASR_COMPUTE", "default")
        _model = WhisperModel(MODEL_FW, device=device, compute_type=compute)
    return _model


def _mlx_model():
    global _model
    if _model is None:
        from mlx_whisper.load_models import load_model
        _model = load_model(MODEL_MLX)
    return _model


def transcribe_words(audio, language=None, robust=False):
    """-> (list of {word, start, end}, detected language code).

    `robust` turns on the temperature fallback ladder, for audio poor enough
    that a dropped segment is the bigger risk than a re-rolled one.
    """
    temps = TEMPERATURES_ROBUST if robust else TEMPERATURES_EXACT
    if backend() == "mlx":
        import mlx_whisper
        res = mlx_whisper.transcribe(
            audio, path_or_hf_repo=MODEL_MLX, word_timestamps=True,
            temperature=temps,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            condition_on_previous_text=False,
            initial_prompt=HOTWORDS or None,
            language=language)
        words = []
        for seg in res["segments"]:
            for w in seg.get("words", []):
                text = w["word"].strip()
                if text:
                    words.append({"word": text,
                                  "start": round(float(w["start"]), 3),
                                  "end": round(float(w["end"]), 3)})
        return words, res.get("language")

    segments, info = _fw_model().transcribe(
        audio, language=language, word_timestamps=True,
        temperature=temps,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOGPROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        condition_on_previous_text=False,
        hotwords=HOTWORDS or None,
        # Silero VAD trims the dead air before decoding. On a call with long
        # silences and a ringback tone that is where whisper drifts.
        vad_filter=True)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if text:
                words.append({"word": text,
                              "start": round(float(w.start), 3),
                              "end": round(float(w.end), 3)})
    return words, info.language


def transcribe_text(audio, language=None):
    """Plain text for one short segment."""
    if backend() == "mlx":
        import mlx_whisper
        res = mlx_whisper.transcribe(
            audio, path_or_hf_repo=MODEL_MLX, language=language,
            temperature=TEMPERATURES_ROBUST, condition_on_previous_text=False)
        return (res.get("text") or "").strip()

    segments, _ = _fw_model().transcribe(
        audio, language=language, temperature=TEMPERATURES_ROBUST,
        condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments).strip()


def language_probs(audio):
    """-> {language code: probability} for one segment."""
    if backend() == "mlx":
        import mlx.core as mx
        from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES
        from mlx_whisper.decoding import detect_language
        model = _mlx_model()
        mel = log_mel_spectrogram(pad_or_trim(mx.array(audio), N_SAMPLES),
                                  n_mels=model.dims.n_mels)
        _, probs = detect_language(model, mel)
        return probs if isinstance(probs, dict) else probs[0]

    _, _, all_probs = _fw_model().detect_language(audio=audio)
    return dict(all_probs)


def model_cached():
    """Has the speech model already been downloaded?

    The weights are about 1.5 GB and are fetched on first use, so the app can
    say whether it is ready to work offline or still has a download ahead of
    it.
    """
    import pathlib
    home = os.environ.get("HF_HOME") or (pathlib.Path.home() / ".cache/huggingface")
    hub = pathlib.Path(home) / "hub"
    if not hub.is_dir():
        return False
    for d in hub.glob("models--*whisper-large-v3-turbo*"):
        if any((d / "snapshots").glob("*/*")):
            return True
    return False


def warm():
    """Load the model now rather than on first use."""
    _fw_model() if backend() == "faster-whisper" else _mlx_model()
