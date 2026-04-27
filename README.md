# tightcut

Automatically remove silences and filler words from a video, locally, on your machine.

`tightcut` is a single Python script that:

1. Extracts audio from a video,
2. Transcribes it with [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) using word-level timestamps,
3. Identifies long pauses and configurable filler words (`ehm`, `uhm`, `cioè`, ...),
4. Cuts those segments out and re-encodes a tighter version of the original video using `ffmpeg`.

No cloud, no API key, no subscription. Designed and tuned for **Italian** out of the box, but works for any language Whisper supports.

---

## Why

Talking-head and screen-recorded videos are full of dead air and verbal tics: "ehm", "uhm", "cioè", "diciamo", and long pauses between sentences. Cutting them by hand is tedious; SaaS tools (Descript, Gling, AutoCut, OpusClip) work but cost money, send your audio to the cloud, and assume English.

`tightcut` does the same thing entirely on your laptop, for free, in the languages Whisper supports.

---

## Features

- **Local-only**: nothing leaves your machine.
- **Cross-platform**: macOS, Linux, Windows. Picks the fastest available backend automatically.
- **Word-level cuts**: removes individual filler words, not just whole pauses.
- **Italian-tuned defaults** (configurable for any language).
- **Dual whisper backend**: [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon (uses GPU + Neural Engine), [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) everywhere else (CPU + CUDA).
- **Cached transcription**: re-run with different thresholds in seconds without re-transcribing.
- **Dry run mode**: preview cuts before encoding.
- **Two filler tiers**: hesitations only (default) vs. hesitations + discourse markers (`--aggressive`).
- **Self-contained**: PEP 723 inline metadata, run with `uv` — no `pip install`, no venv to manage.

---

## Requirements

- **Python 3.11+** (handled automatically by `uv`).
- **`ffmpeg`** (>= 8.0 recommended) and **`uv`** on `PATH`.
- **~3 GB disk** for the Whisper model on first run (cached afterwards in `~/.cache/huggingface`).
- **Apple Silicon Mac** for the fastest path (uses MLX). Linux/Windows/Intel Macs work via the `faster-whisper` fallback.

### Install

**macOS:**
```bash
brew install ffmpeg uv
```

**Linux:**
```bash
# Debian/Ubuntu
sudo apt install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:** install [ffmpeg](https://ffmpeg.org/download.html) and [uv](https://docs.astral.sh/uv/getting-started/installation/) from their official pages.

For the encoder, defaults are: `h264_videotoolbox` on macOS, `libx264` everywhere else (override with `--encoder`).

---

## Usage

```bash
# Basic: cuts hesitations and silences > 0.5s
uv run --script tightcut.py input.mov

# Output goes to input.cut.mov by default. Override with -o:
uv run --script tightcut.py input.mov -o edited.mov

# See what would be cut, without actually re-encoding
uv run --script tightcut.py input.mov --dry-run

# More aggressive: also drops discourse markers (cioè, tipo, diciamo, ...)
uv run --script tightcut.py input.mov --aggressive

# Tighter pacing
uv run --script tightcut.py input.mov --max-silence 0.3 --pad 0.05

# Silence only, leave fillers in
uv run --script tightcut.py input.mov --no-fillers

# Slower but more accurate model
uv run --script tightcut.py input.mov --model large-v3

# Other languages
uv run --script tightcut.py talk.mp4 --language en
uv run --script tightcut.py charla.mp4 --language es
```

### All options

| Flag | Default | What it does |
| --- | --- | --- |
| `input` | required | Path to input video (`.mov`, `.mp4`, anything ffmpeg reads). |
| `-o, --output` | `<input>.cut.mov` | Output path. |
| `--language` | `it` | Whisper language code (`it`, `en`, `es`, ...). |
| `--model` | `large-v3-turbo` | `tiny` / `base` / `small` / `medium` / `large-v3` / `large-v3-turbo`. |
| `--max-silence` | `0.5` | Silences longer than this (seconds) are cut. |
| `--pad` | `0.08` | Padding kept on each side of a cut, to avoid clipping speech. |
| `--aggressive` | off | Also cut discourse markers (see [Filler word lists](#filler-word-lists)). |
| `--no-fillers` | off | Cut silences only, keep all words. |
| `--backend` | `auto` | `auto` / `mlx` (Apple Silicon, GPU+ANE) / `faster-whisper` (everywhere, CPU+CUDA). |
| `--encoder` | `auto` | `auto` / `h264_videotoolbox` (macOS) / `libx264` (portable) / `h264_nvenc` (NVIDIA). |
| `--no-cache` | off | Ignore the cached transcription and re-run Whisper. |
| `--dry-run` | off | Print the planned cuts and exit, no re-encoding. |

---

## How it works

```
input.mov
   |
   |  ffmpeg: extract 16 kHz mono PCM
   v
audio.wav  ----> whisper (large-v3-turbo, word_timestamps=True)
                    |
                    | backend = auto-detect:
                    |   Apple Silicon  -> mlx-whisper (GPU + Neural Engine)
                    |   everywhere else -> faster-whisper (CPU, optionally CUDA)
                    v
              [{word, start, end}, ...]   <-- cached as input.mov.whisper.json
                    |
                    |  build cuts:
                    |    - gap between words > max-silence -> cut
                    |    - normalized word in filler set    -> cut
                    v
              cuts: [(start, end, reason), ...]
                    |
                    |  invert -> keeps
                    v
              keeps: [(start, end), ...]
                    |
                    |  ffmpeg: filter_complex with trim/atrim/concat,
                    |          re-encode via h264_videotoolbox / libx264 / h264_nvenc
                    v
               output.cut.mov
```

### Why these choices

- **Two whisper backends.** `mlx-whisper` runs the model on the Apple GPU + Neural Engine via Apple's MLX framework — typically 2–3× faster than CPU-only inference on M-series chips, thanks to unified memory. Everywhere else, `faster-whisper` (CTranslate2, `int8` quantized) is the most portable fast option, runs on CPU or CUDA, and supports Linux/Windows/Intel Macs. The script detects the platform at startup and picks the right one; `--backend` overrides if you want.
- **`large-v3-turbo` as default**: ~6× faster than `large-v3` with marginal quality loss. For a 30-minute Italian recording this saves ~10 minutes per run. Use `--model large-v3` if you need maximum accuracy.
- **Verbatim prompt + `condition_on_previous_text=False`**: vanilla Whisper is trained to *strip* disfluencies. The initial prompt nudges it toward verbatim transcription so we get a chance to detect "ehm" / "uhm" as actual tokens.
- **VAD pre-filter** (only available on `faster-whisper`): segments audio at silences before transcription, helping reduce hallucinations on long pauses.
- **`filter_complex` with `trim` + `atrim` + `concat`**: produces frame-accurate cuts. Stream copy can't cut inside a GOP, so re-encoding is unavoidable. Hardware encoding (VideoToolbox / NVENC) keeps it fast.
- **Transcription cache**: lets you tune `--max-silence` / `--aggressive` / `--pad` interactively without paying the transcription cost every time.

---

## Filler word lists

Defined at the top of [`tightcut.py`](./tightcut.py).

**Hesitations** (default; almost never legitimate words):

```
ehm, ehmm, ehmmm, uhm, uhmm, mh, mhm, mmm, mmmm, mmh,
hmm, eh, ehh, ah, ahh, uh, uhh, eee, ee, umm
```

**Discourse markers** (only with `--aggressive`; can be legitimate, use carefully):

```
cioè, tipo, diciamo, praticamente, insomma, boh,
ecco, appunto, comunque, allora
```

Edit the `HESITATION_FILLERS` and `DISCOURSE_FILLERS` sets in the script to customize for your language or speaking style.

---

## Performance

Measured on an Apple Silicon Mac (M-series), 30 fps 1080p H.264 input:

| Stage | Throughput | Notes |
| --- | --- | --- |
| Audio extraction | ~1000× realtime | Trivial. |
| Transcription (`mlx-whisper`, `large-v3-turbo`) | **~3.5× realtime** | GPU + Neural Engine via MLX. |
| Transcription (`faster-whisper`, `large-v3-turbo`, int8 CPU) | ~1.8× realtime | Portable fallback. |
| Transcription (`faster-whisper`, `large-v3`, int8 CPU) | ~0.5× realtime | Higher quality. |
| Re-encoding (`h264_videotoolbox`, 12 Mbps) | ~7× realtime | Hardware-accelerated. |

For a 32-minute recording on Apple Silicon (mlx-whisper + VideoToolbox): expect about **9 min** transcription + **5 min** encoding on first run. On a re-run with cached transcription and just a threshold change: **~5 min total** (encoding only).

---

## Limitations

- **Re-encoding is mandatory.** Cuts at non-keyframe boundaries can't use stream copy. Quality is preserved at sensible bitrates but it's not bit-identical to the source.
- **Filler word detection is imperfect.** Whisper drops some "ehm"s no matter how you prompt it. Silence cuts catch most of them anyway because hesitations are usually flanked by pauses.
- **Default bitrate is 12 Mbps for the VideoToolbox encoder.** If your source is a high-bitrate ProRes or H.265 master, edit the `assemble()` function or use `--encoder libx264` with CRF.
- **Stereo is preserved but spatial information may drift slightly** at cut boundaries because audio is re-encoded as AAC.
- **Speaker diarization is not implemented**: cuts apply uniformly. Multi-speaker interview content with overlapping speech may need manual review.
- **No GUI.** This is a CLI tool. If you want a timeline editor with previews, use Descript, Gling, or auto-editor.

---

## Suggested workflow

1. Run with `--dry-run` first to verify the cuts look reasonable.
2. If too many words are being cut, increase `--max-silence` or remove `--aggressive`.
3. If too much dead air remains, decrease `--max-silence`.
4. Once happy with the dry-run output, re-run without `--dry-run` to produce the final file. The transcription cache makes iterations cheap.
5. Always do a final manual review pass — these tools are time-savers, not replacements for editorial judgement.

---

## Roadmap ideas

- [ ] Custom filler list via CLI flag instead of editing the file
- [ ] Export an EDL / FCP7 XML / DaVinci Resolve timeline instead of a flattened video
- [ ] Speaker diarization to skip cuts on the secondary speaker
- [ ] Loudness normalization (`loudnorm`) as a post-processing pass
- [ ] VAD pre-filter for the `mlx-whisper` backend (currently only on `faster-whisper`)
- [ ] Tests

---

## Credits

- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — portable speech recognition backend (CPU + CUDA)
- [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple Silicon backend, built on Apple's [MLX](https://github.com/ml-explore/mlx)
- [OpenAI Whisper](https://github.com/openai/whisper) — the underlying model
- [`ffmpeg`](https://ffmpeg.org/) — for everything video-related
- [`uv`](https://github.com/astral-sh/uv) — for friction-free Python script execution

---

## License

MIT.
