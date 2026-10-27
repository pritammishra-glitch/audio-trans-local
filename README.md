# Local Audio Transcription & Annotation Tool

A machine pre-annotation pipeline for the Audio Transcription and Annotation
Guidelines. It fills what can be measured or reliably inferred, and marks the
rest for the reviewer rather than guessing.

## Requirements

| | |
|---|---|
| Machine | Apple Silicon Mac (M1-M5), 16 GB RAM |
| Python | 3.9 or newer |
| Ollama | running locally |
| Disk | about 7 GB for the two models |
| Internet | only on first run, to download the models |

### Platform support

| Platform | Speech engine | Status |
|---|---|---|
| macOS, Apple Silicon | mlx-whisper, on the GPU | **Supported.** Built and tested, `.dmg` available |
| Windows | faster-whisper, CPU or CUDA | **Supported.** Code tested; build it with `build_windows.bat` |
| Linux, Intel Mac | faster-whisper, CPU or CUDA | Same as Windows |

Only the speech engine differs, and it sits behind `src/asr_backend.py`, so
every other stage is identical on all three. mlx-whisper is much faster but
exists only for Apple Silicon; faster-whisper runs anywhere.

## Setup on macOS (Apple Silicon)

No Homebrew required. The system Python that ships with the Xcode command line
tools is enough.

**1. Command line tools**, if `python3 --version` fails:

```bash
xcode-select --install
```

**2. Ollama** - download from [ollama.com/download](https://ollama.com/download),
or if you do have Homebrew:

```bash
brew install --cask ollama
```

Then start it and pull the annotation model (about 4.7 GB):

```bash
ollama serve &
ollama pull qwen2.5:14b-instruct-q4_K_M   # or qwen2.5:7b for half the time
```

The annotation model is picked automatically from what is installed, best
first. Measured on five real turns, same prompt, same machine:

| Model | Speed | Verbatim + clean | `<fs>`/`<sc>` found |
|---|---|---|---|
| qwen2.5:14b | 6.2 s/turn | 5/5 | 3 of 3 |
| qwen2.5:7b | 3.0 s/turn | 4/5 | 1 of 3 |
| qwen3:8b | 30.8 s/turn | 5/5 | 3 of 3 |

qwen3 is a reasoning model and spends ten times as long to reach the same
answer, so it is not used. `OLLAMA_MODEL` overrides the choice.

Turns are annotated eight at a time rather than one by one, which stopped the
900-token rulebook being re-read for every turn: on a nine-minute call that
took the annotation pass from about 25 minutes to 11. `ANNOTATE_BATCH=1`
restores one-at-a-time if a batch ever misbehaves - any batch that comes back
the wrong shape already falls back by itself.

**3. The Python environment**, from inside the project folder:

```bash
cd "/path/to/audi trans"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

**4. Stop Ollama pinning memory between files.** It keeps a model resident for
five minutes after each call by default, which leaves 5.5 GB held while the next
file is trying to transcribe:

```bash
echo 'export OLLAMA_KEEP_ALIVE=0' >> ~/.zshrc
source ~/.zshrc
```

`run.sh` sets this for its own run too, so this step only matters if you call
the stages directly.

That is the whole install. The whisper weights (about 1.5 GB) download by
themselves the first time you transcribe something, so the first run is slower
than the rest.

## Running it

```bash
cd "/path/to/audi trans"
./serve.sh
```

Then open **http://127.0.0.1:5001**. Drag in a `.wav`, `.mp3`, `.m4a`, `.flac`
or `.caf`, choose the languages, and the page hands you **transcript.txt** to
download when it finishes. `Ctrl+C` stops the server.

Nothing leaves the machine: the server binds to 127.0.0.1 and both models run
locally.

The packaged app (see below) behaves like an application instead: it opens its
own window, picks a free port by itself, and closing the window quits it and
stops any transcription still running. `src/app.py` does the same thing when
run from source; set `NO_WINDOW=1` to get the browser behaviour instead.

### From the command line instead

```bash
./run.sh audio/your_call.wav              # auto-detect language
./run.sh audio/your_call.wav en           # force the main language
./run.sh audio/your_call.wav en kn        # ...and name the second language
FAST=1 ./run.sh audio/your_call.wav en    # skip the LLM pass
```

### Checking it works

```bash
./.venv/bin/python src/make_test_call.py audio/test.wav bank
./run.sh audio/test.wav en
cat out/test/transcript.txt
```

That builds a synthetic two-speaker call with macOS's own voices and runs it
end to end, so you can confirm the install without supplying a recording.

### If something goes wrong

| Symptom | Cause |
|---|---|
| `ollama: connection refused` | Ollama is not running. `ollama serve &` |
| Everything crawls mid-run | `OLLAMA_KEEP_ALIVE=0` is not set, so 5.5 GB stays pinned |
| `mono input -> stereo split unavailable` | Normal for a mono file; it falls back to clustering |
| Stage 6 takes minutes | Expected - it is one model call per turn. Use fast mode |
| First run hangs at stage 1 | It is downloading the 1.5 GB whisper weights |

## Shipping it to someone else

### The .dmg (macOS, Apple Silicon)

```bash
./.venv/bin/pip install -r requirements-build.txt
./build_macos.sh
```

That produces `dist/AudioTranscribe-1.0.0-AppleSilicon.dmg`, about 137 MB. It
holds the app, a shortcut to Applications and a FIRST-RUN note. The recipient
drags it across and double-clicks it - it opens in its own window, with no
setup, no terminal and no configuration. Closing the window quits it and takes
any running stage process with it.

It works on a machine with nothing else installed: with no Ollama, the page
says so and runs in Fast mode, which needs no model beyond whisper. Installing
Ollama later and reloading the page unlocks the full annotation pass, with no
reinstall.

Two things to tell whoever you send it to:

1. **Gatekeeper will block the first launch**, because the app is not signed
   with an Apple Developer ID. They right-click it and choose *Open*, once. If
   macOS calls it "damaged", `xattr -cr /Applications/AudioTranscribe.app`
   clears the quarantine flag. Signing and notarising removes this entirely
   and needs a paid Apple Developer account.
2. **Apple Silicon only.** An Intel Mac cannot run it, for the MLX reason
   above.
   (Fast mode versus full mode is handled by the app itself - it detects
   Ollama and explains the difference on the page.)

The first transcription downloads the whisper weights (about 1.5 GB) into
`~/.cache/huggingface`, so it is slower than the ones after it. Results are
written to `~/Library/Application Support/AudioTranscribe/out/`, not next to
the app, because an app bundle is read-only.

How the packaging works, in case it needs changing: `src/app.py` is the entry
point and does two jobs - started plainly it serves the page, started as
`--stage <name>` it runs one pipeline stage. `src/pipeline.py` drives the
sequence by re-invoking the binary once per stage, which is how the bundle
keeps the one-model-at-a-time memory design without a shell script. Two
details earned their comments the hard way: `multiprocessing.freeze_support()`
has to be the first thing `app.py` runs, or the helper process huggingface_hub
starts during a download re-enters the app and hangs it; and `mlx` and
`mlx_whisper` need `collect_all`, because their Metal libraries and mel filter
bank are invisible to static analysis.

### The .exe (Windows)

**PyInstaller cannot cross-compile, so this has to be built on a Windows
machine.** Everything else is done - the code is ported and the build is one
command:

```bat
build_windows.bat
```

That creates the virtual environment, installs the dependencies, checks the
pipeline imports, and writes `dist\AudioTranscribe\AudioTranscribe.exe`.
Zip the whole `dist\AudioTranscribe` folder to share it; the .exe needs the
files beside it.

Tell whoever you send it to:

1. **Keep the folder together** - the .exe alone will not run.
2. **SmartScreen will warn** that the publisher is unknown, because the build
   is not code-signed. *More info* then *Run anyway*, once.
3. **Fast mode works on its own; full mode needs Ollama**, same as the Mac
   build.
4. **.mp3 and .m4a need ffmpeg** on PATH (`winget install Gyan.FFmpeg`).
   `.wav` and `.flac` are read directly.

Expect roughly real time on a CPU - a 10 minute call takes about 10 minutes,
against 1 minute on a Mac GPU. With an NVIDIA card, set `ASR_DEVICE=cuda` and
`ASR_COMPUTE=float16` before launching.

Results go to `%LOCALAPPDATA%\AudioTranscribe\out\`.

**What has and has not been tested.** The faster-whisper backend was run end
to end on real audio, and a bundle built from
`AudioTranscribe-windows.spec` was frozen and run successfully - both on a
Mac, with the backend forced. What could not be tested from here is Windows
itself: its DLL packaging, SmartScreen, and paths. The first Windows build
may need small fixes.

## Windows, Linux and Intel Macs

These run the same pipeline with a different speech engine.
`src/asr_backend.py` holds the two implementations behind one interface:

| | Engine | Speed |
|---|---|---|
| Apple Silicon | mlx-whisper, on the Mac GPU | about 0.1x real time |
| Everything else | faster-whisper (CTranslate2), CPU or NVIDIA GPU | about 1x real time on CPU |

The choice is automatic - mlx if it imports, faster-whisper otherwise. Set
`ASR_BACKEND=mlx` or `ASR_BACKEND=faster-whisper` to force one, which is how
the faster-whisper path is tested on a Mac.

### Setup

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements-windows.txt
winget install Gyan.FFmpeg
```

Then Ollama from [ollama.com/download](https://ollama.com/download), and:

```bat
ollama pull qwen2.5:7b
setx OLLAMA_KEEP_ALIVE 0
```

### Running

`run.sh` and `serve.sh` are bash. On Windows use the Python entry point
instead, which does the same thing:

```bat
.venv\Scripts\python src\app.py
```

That serves the page at http://127.0.0.1:5001 and opens a browser. For a
single file without the UI:

```bat
.venv\Scripts\python -c "import sys; sys.path.insert(0,'src'); import pipeline; pipeline.run('audio/call.wav','out/call',language='en',second='kn',fast=True)"
```

An Intel Mac follows the same steps, except that `afconvert` is already
present so ffmpeg is optional, and `run.sh` and `serve.sh` work as written.

## Output

Two files land in `out/<filename>/`, with every stage's JSON beside them so you
can rerun one stage without redoing the others.

**`transcript.txt`** - one line per turn, for delivery:

```
[00:16.120 - 00:21.220] Speaker B: Uh yeah, actually I have made one appointment at around <TIME>3 .30</TIME>.
[00:28.700 - 00:31.240] Speaker A: Yeah, <DATE>today</DATE> itself, correct?
[00:10.140 - 00:12.960] Speaker B: Actually, I am from <LOCATION>Bangalore</LOCATION>.
```

Speakers are lettered A, B, C... in order of first appearance. Privacy tags are
written in the same bare upper-case style as the entity tags
(`<NAME>`, `<CARD>`, `<OTP>`, `<ACCOUNT>`), not the guidelines' `<pii:...>`
form.

**`transcript_review.txt`** - the guidelines' two-line As spoken / Clean
format, with the metadata block and the `>> REVIEW` flags that tell a reviewer
where to listen.

### Other languages

A turn not in the call's main language is wrapped by name:

```
[00:02.900 - 00:05.340] Speaker B: <Kannada>ನಮಸ್ಕಾರ ನಾನು ವೈದ್ಯರನ್ನು ನೋಡಬೇಕು.</Kannada>
[00:09.200 - 00:11.140] Speaker B: <Kannada>[unclear]</Kannada>
```

Detecting *that* a turn is not the main language is reliable. Naming *which*
language it is, is not, and the measurements are worth stating because they
drove the design:

- On 8 kHz telephone English the detector returned Spanish at 0.78 and
  Indonesian at 0.87. Re-transcribing did not settle it either - Spanish beat
  English on log-probability for a turn that plainly says "I don't know".
- On clean Kannada it answered Hindi 0.33 / Tamil 0.32 and never offered
  Kannada at all. Pooling 6.5 s of it together still only reached Hindi 0.29.

So the pipeline does three things instead of guessing. It ignores Latin-script
candidates, where the detector was shown to be wrong. It pools all the
suspected non-main-language audio and identifies it once, rather than from
two-second clips. And it only writes a language name when that pooled answer
clears 0.50 - otherwise the turn is tagged `<OtherLanguage>` with the
candidates listed on the review line.

**Naming the second language in the UI removes all of that guesswork**, and is
worth doing when you know what it is. The turn is then re-transcribed in that
language. If the re-transcription collapses into a repetition loop - which
whisper does on short clips - the turn becomes `<Kannada>[unclear]</Kannada>`
rather than passing the loop off as content.

Whisper also sometimes transliterates an Indian language into Latin script by
itself ("Namaskara, Nanu Vaidya Ranu Noda Beku"), which is what the guidelines
ask for. Where it emits a non-Latin script instead, the script alone names the
language with certainty and no detection is needed.

### Fast mode

`FAST=1 ./run.sh ...`, or the checkbox in the UI. It skips the language-model
pass, which is most of the runtime: on a 9-minute call stage 6 went from about
6 minutes to 0.2 seconds.

You keep timestamps, speaker labels, pauses, silences, interrupts, and the
PII, `<DATE>` and `<TIME>` tags - all of which are produced without a model.
You lose the Clean line, `<fs>`/`<sc>` and `<LOCATION>`.

### Entity tags

`<DATE>` and `<TIME>` are tagged by regex in [src/entities.py](src/entities.py)
- including the forms the ASR actually produces, like "3 .30" and
"5 o'clock pm" - and are guarded so they cannot swallow the decimal part of a
currency amount. `<LOCATION>` gets its own small model call, asked only when
the turn holds a capitalised word that could be a place; buried in the main
annotation prompt it was missed every time.

## How this build differs from the guide

The guide assumes Homebrew, Python 3.11, a compiled whisper.cpp and
TensorFlow. None of that was present, and all of it was avoidable:

| Guide | This build | Why |
|---|---|---|
| whisper.cpp built with cmake | `mlx-whisper`, same large-v3-turbo weights | Metal-native, pip-installable, no Xcode build |
| `ffmpeg` for audio prep | `stage0_prep.py` (soundfile + scipy) | ffmpeg is not installed; CoreAudio's `afconvert` decodes anything else |
| Python 3.11 venv | system Python 3.9 venv | every wheel needed has a cp39 arm64 build |
| SpeechBrain ECAPA for gender | autocorrelation pitch | the guide's own code only used pitch; ECAPA was never read |
| YAMNet for the 17 noise labels | measured packet-drop + steady-noise candidates | TensorFlow has no working toolchain here; see "Not yet built" |
| `qwen2.5:7b-instruct-q4_K_M` | `qwen2.5:7b` | already pulled locally; same q4_K_M quantisation |
| one pass over the mixed mono track | one pass per channel on stereo input | whisper hallucinates over overlapping voices, and the channel then names the speaker exactly |

### Bugs fixed against the guide's listings

- `afconvert -c 1` **discards** a channel rather than summing it, which
  silently lost one speaker entirely. The downmix is done in numpy.
- `speaker_at()` matched a word only if its start fell inside a turn. Whisper's
  word times and energy-derived turn edges disagree by a fraction of a second,
  so every other word came out `UNKNOWN` and each word became its own turn.
  It now matches on maximum overlap, with a nearest-turn fallback.
- The channel activity mask fragmented at every sub-0.5 s gap. It is now
  bridged at 0.5 s, which is the same threshold the spec uses to start a new
  line.
- Inserting several pauses into one turn used stale word indices, so the second
  and later pauses landed in the wrong place.
- The packet-drop window of 40–200 ms is shorter than an ordinary stop
  consonant closure. On clean audio it produced 18 false candidates; requiring
  ≥100 ms, a collapse all the way to the channel noise floor, and a position
  strictly inside a transcribed word brought that to 0 while still catching an
  injected 150 ms dropout.

### Guards added in the annotation layer

Three failures were silent enough that a reviewer could miss them, so each gets
one corrective retry and then a loud flag:

- **Digit corruption.** The model turned `9014` into `9814` in a Clean line.
  Digits in Clean that never appeared in As spoken are now detected
  (`digits_differ_from_as_spoken_CHECK`).
- **Emptied Clean lines.** The backchannel rule was over-applied to turns
  carrying real content (`clean_line_empty_CHECK`).
- **Missed PII.** Deterministic rules now tag `Mr/Ms <Name>`, four digits after
  "card"/"account", codes after "code"/"OTP"/"PIN", emails, 10-digit phones and
  12–19 digit card numbers, whatever the model did. A trigger word opens a
  window and *every* digit group inside it is tagged, because callers read a
  number back twice and only tagging the first copy leaves the second exposed.
  Currency amounts are excluded, so `$1,284.37` is never mistaken for an
  account number.
- **Altered measurements.** The model was seen replacing a measured
  `[pause: 0.7s]` with an invented `[unclear]`. Pause markers must now survive
  the annotation pass unchanged, and `[unclear]` may not be introduced where
  the input had none (`pause_markers_altered_CHECK`, `unclear_invented_CHECK`).

`possible_hesitation` is also gated: the model proposed it on almost every
turn, so it is kept only where a pause was actually measured. A full card
number spoken with a CVV halts stage 6 rather than warning, as the spec
requires.

## What it fills, and what it refuses to

Filled: words and timestamps, speaker labels, `[pause: N.Ns]` and
`[silence: N.Ns]` (measured off the waveform), `[interrupt]` from channel
overlap, gender from pitch, PII tags, the Clean line, and the flag-the-file
pre-checks.

Deliberately left for the reviewer, as `[REVIEW]`, `unclear`, `needs_review`
or a `>> REVIEW:` line: the scenario, emotional state, accent, and every
`<sarcasm>`, `<mispron>`, `[pressure]` and `[hesitation]` judgement.

## Known limitations

- **Whisper normalises numbers.** It writes "9014" where the speaker said
  "nine zero one four", so the As spoken line is not strictly verbatim for
  numbers. It has also mis-heard "ninety fourteen" as "1914". Numbers must be
  checked by ear regardless — the spec requires them to be 100% correct.
- **Whisper still repeats itself occasionally** on a long turn, emitting a
  paraphrase of something said earlier. Per-channel transcription made this
  much rarer but did not remove it.
- `<fs>` and `<sc>` tagging is inconsistent at 7B; the guide rates it moderate.
  A 14B model on a 24 GB machine is the recommended upgrade.
- Gender sits in the 145–180 Hz unclear band for some female voices. That band
  is deliberate, not a bug — it is where classifiers go wrong.
- Mono input is rejected by stage 2. The pyannote fallback in the guide needs a
  Hugging Face token and is not installed.

## Not yet built

- **YAMNet** (`[noise: horn]` and the 17-label set) — needs
  `tensorflow-macos` + `tensorflow-hub`. Stage 4 currently emits only
  measurable candidates and never asserts a `[noise: ...]` tag.
- **pyannote diarization** for mono recordings. The guide's fallback needs a
  Hugging Face token and a licence accepted in a browser, so mono files use a
  local stand-in instead: segment on the word timings, describe each segment
  with MFCCs and pitch, cluster into two speakers. It is weaker than neural
  diarization and every mono run says so on a `!!` line.
- **`[bg_speech]`** separation, which the guide notes needs stage 2 channel
  data combined with a speech classifier.

## Layout

```
audio/              input files
out/<name>/         original.wav, mono16k.wav, every stage's JSON, transcript.txt
src/stage0_prep.py      decode, downmix, resample
src/stage1_asr.py       words + per-word timestamps (mlx-whisper)
src/stage2_speakers.py  stereo channel split, overlaps, empty-track check
src/stage3_acoustics.py pauses, silences, noise floor
src/stage4_events.py    packet-drop and steady-noise candidates
src/stage5_speaker_meta.py  gender from pitch; accent/emotion left unfilled
src/stage7_merge.py     words + speakers + pauses -> turns (runs before stage 6)
src/stage6_annotate.py  the Ollama rule layer + PII guards
src/stage8_format.py    both output files
src/entities.py         <DATE> / <TIME> tagging
src/stage2_mono.py      mono fallback: MFCC + pitch clustering
src/stage7b_language.py per-turn language detection and tagging
src/webapp.py           the upload-and-download front end
src/precheck.py         the flag-the-file rules
src/make_test_call.py   builds the synthetic stereo test call
src/pipeline.py         stage orchestration, works frozen or from source
src/app.py              packaged-app entry point
AudioTranscribe.spec    PyInstaller build definition
build_macos.sh          builds the .app and the .dmg
requirements.txt        runtime dependencies
requirements-build.txt  the above plus PyInstaller
```

