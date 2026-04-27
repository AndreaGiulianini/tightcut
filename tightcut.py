#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "faster-whisper>=1.1.0",
#     "mlx-whisper>=0.4.2; platform_system == 'Darwin' and platform_machine == 'arm64'",
#     "tqdm>=4.66",
# ]
# ///
"""tightcut: remove silences and filler words from a video using whisper + ffmpeg.

Picks the fastest available backend automatically:
  - mlx-whisper on Apple Silicon (uses GPU + Neural Engine via MLX)
  - faster-whisper everywhere else (Linux / Windows / Intel Mac, CPU + CUDA)

Examples:
    ./tightcut.py input.mov
    ./tightcut.py input.mov -o out.mov --max-silence 0.4 --aggressive
    ./tightcut.py input.mov --dry-run            # see what would be cut
    ./tightcut.py input.mov --model large-v3     # max accuracy (slower)
    ./tightcut.py input.mov --backend faster-whisper   # force CPU backend
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

from tqdm import tqdm


HESITATION_FILLERS = {
    "ehm", "ehmm", "ehmmm", "uhm", "uhmm", "mh", "mhm", "mmm", "mmmm", "mmh",
    "hmm", "eh", "ehh", "ah", "ahh", "uh", "uhh", "eee", "ee", "uhh", "umm",
}

DISCOURSE_FILLERS = {
    "cioè", "tipo", "diciamo", "praticamente", "insomma", "boh",
    "ecco", "appunto", "comunque", "allora",
}

# user-facing model size -> mlx-community HF repo
MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

VERBATIM_PROMPT_IT = (
    "Trascrizione fedele parola per parola, "
    "includendo esitazioni come ehm, uhm, eh, ah, mh."
)


@dataclass
class Word:
    text: str
    start: float
    end: float


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True)
    return float(out.strip())


def extract_audio(video: Path, wav: Path) -> None:
    print(f"[*] Extracting audio to {wav.name} (16 kHz mono PCM)...")
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-stats", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav),
    ])


def detect_backend() -> str:
    """Pick the fastest backend available on this machine."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "faster-whisper"


def detect_encoder() -> str:
    """Pick the best portable hardware encoder for this machine."""
    if platform.system() == "Darwin":
        return "h264_videotoolbox"
    return "libx264"


def transcribe(wav: Path, language: str, model_size: str, backend: str) -> list[Word]:
    if backend == "auto":
        backend = detect_backend()
    print(f"[*] Backend: {backend}  |  Model: {model_size}  |  Language: {language}")
    if backend == "mlx":
        return _transcribe_mlx(wav, language, model_size)
    return _transcribe_faster_whisper(wav, language, model_size)


def _transcribe_mlx(wav: Path, language: str, model_size: str) -> list[Word]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise SystemExit(
            "mlx-whisper not installed. It only runs on Apple Silicon. "
            "Use --backend faster-whisper to fall back, or run on a Mac."
        ) from exc
    repo = MLX_MODEL_MAP.get(model_size, model_size)
    print(f"[*] Loading mlx-whisper '{repo}' (first run downloads it from HF)...")
    print("[*] Transcribing on Apple GPU/Neural Engine...")
    result = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=repo,
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=VERBATIM_PROMPT_IT if language == "it" else None,
        verbose=False,
    )
    words: list[Word] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            text = w.get("word") or w.get("text") or ""
            if not text.strip():
                continue
            words.append(Word(text.strip(), float(w["start"]), float(w["end"])))
    return words


def _transcribe_faster_whisper(wav: Path, language: str, model_size: str) -> list[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("faster-whisper not installed. Reinstall the script's dependencies.") from exc
    print(f"[*] Loading faster-whisper '{model_size}' (first run downloads it)...")
    # int8 CPU is the most portable path; on CUDA you can edit this to ("cuda", "float16").
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("[*] Transcribing on CPU...")
    segments, info = model.transcribe(
        str(wav),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
        beam_size=5,
        initial_prompt=VERBATIM_PROMPT_IT if language == "it" else None,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    pbar = tqdm(total=round(info.duration, 2), unit="s",
                desc="transcribe", smoothing=0.05, bar_format="{l_bar}{bar}{r_bar}")
    last = 0.0
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append(Word(w.word.strip(), float(w.start), float(w.end)))
        pbar.update(max(0.0, seg.end - last))
        last = seg.end
    pbar.close()
    return words


def normalize_token(s: str) -> str:
    return s.strip(" .,?!\"'…-—–:;()").lower()


def build_cuts(
    words: list[Word],
    duration: float,
    fillers: set[str],
    max_silence: float,
    pad: float,
) -> list[tuple[float, float, str]]:
    """Return list of (start, end, reason) intervals to REMOVE."""
    cuts: list[tuple[float, float, str]] = []
    prev_end = 0.0
    for w in words:
        gap = w.start - prev_end
        if gap > max_silence:
            s = prev_end + pad
            e = w.start - pad
            if e - s > 0.05:
                cuts.append((s, e, "silence"))
        if normalize_token(w.text) in fillers:
            cuts.append((max(0.0, w.start - pad / 2),
                         min(duration, w.end + pad / 2),
                         f"filler:{w.text.strip()}"))
        prev_end = max(prev_end, w.end)
    if duration - prev_end > max_silence:
        cuts.append((prev_end + pad, duration, "silence"))
    return merge(sorted(cuts, key=lambda x: x[0]))


def merge(intervals: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    if not intervals:
        return []
    out = [intervals[0]]
    for s, e, r in intervals[1:]:
        ps, pe, pr = out[-1]
        if s <= pe + 0.01:
            out[-1] = (ps, max(pe, e), pr if pr == r else "mixed")
        else:
            out.append((s, e, r))
    return out


def invert(cuts: list[tuple[float, float, str]], duration: float) -> list[tuple[float, float]]:
    keeps: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e, _ in cuts:
        if s > cursor:
            keeps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        keeps.append((cursor, duration))
    return [(s, e) for s, e in keeps if e - s > 0.05]


def assemble(
    video: Path,
    keeps: list[tuple[float, float]],
    output: Path,
    encoder: str,
) -> None:
    if not keeps:
        raise SystemExit("No segments to keep -- everything looked like silence/fillers.")
    print(f"[*] Assembling {len(keeps)} kept segments -> {output.name}")
    parts: list[str] = []
    for i, (s, e) in enumerate(keeps):
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
    chain = "".join(f"[v{i}][a{i}]" for i in range(len(keeps)))
    parts.append(f"{chain}concat=n={len(keeps)}:v=1:a=1[outv][outa]")

    script_path = output.with_suffix(".filter.txt")
    script_path.write_text(";\n".join(parts))

    cmd = [
        "ffmpeg", "-y", "-stats", "-loglevel", "error",
        "-i", str(video),
        "-filter_complex_script", str(script_path),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", encoder,
    ]
    if encoder == "h264_videotoolbox":
        cmd += ["-b:v", "12M", "-pix_fmt", "yuv420p"]
    elif encoder == "h264_nvenc":
        cmd += ["-cq", "22", "-preset", "p5", "-pix_fmt", "yuv420p"]
    else:  # libx264
        cmd += ["-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
    try:
        run(cmd)
    finally:
        script_path.unlink(missing_ok=True)


def fmt(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - 60 * (h * 60 + m)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Remove silences and filler words from a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--language", default="it")
    p.add_argument("--model", default="large-v3-turbo",
                   help="whisper model: tiny|base|small|medium|large-v3|large-v3-turbo")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "mlx", "faster-whisper"],
                   help="speech recognition backend [auto picks mlx on Apple Silicon, faster-whisper elsewhere]")
    p.add_argument("--max-silence", type=float, default=0.5,
                   help="silences longer than this (s) are cut [default: 0.5]")
    p.add_argument("--pad", type=float, default=0.08,
                   help="padding kept around each cut (s) [default: 0.08]")
    p.add_argument("--aggressive", action="store_true",
                   help="also cut discourse markers: cioè, tipo, diciamo, ...")
    p.add_argument("--no-fillers", action="store_true",
                   help="only cut silences, keep filler words")
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "h264_videotoolbox", "libx264", "h264_nvenc"],
                   help="auto picks h264_videotoolbox on macOS, libx264 elsewhere")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore cached transcription and re-run whisper")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be cut, skip the encode")
    args = p.parse_args()

    video = args.input.resolve()
    if not video.exists():
        sys.exit(f"input not found: {video}")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not on PATH -- run: brew install ffmpeg")

    output = args.output or video.with_name(video.stem + ".cut.mov")
    cache_path = video.with_suffix(video.suffix + ".whisper.json")

    duration = probe_duration(video)
    print(f"[*] Input: {video.name} ({fmt(duration)})")

    fillers: set[str] = set()
    if not args.no_fillers:
        fillers |= HESITATION_FILLERS
        if args.aggressive:
            fillers |= DISCOURSE_FILLERS

    if cache_path.exists() and not args.no_cache:
        print(f"[*] Reusing cached transcription: {cache_path.name} (use --no-cache to redo)")
        words = [Word(**w) for w in json.loads(cache_path.read_text())]
    else:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio.wav"
            extract_audio(video, wav)
            words = transcribe(wav, args.language, args.model, args.backend)
        cache_path.write_text(json.dumps([asdict(w) for w in words], ensure_ascii=False))
        print(f"[*] Cached transcription -> {cache_path.name}")

    cuts = build_cuts(words, duration, fillers, args.max_silence, args.pad)
    keeps = invert(cuts, duration)
    cut_total = sum(e - s for s, e, _ in cuts)
    keep_total = sum(e - s for s, e in keeps)
    n_silence = sum(1 for *_, r in cuts if r == "silence" or r == "mixed")
    n_filler = len(cuts) - n_silence
    print(f"[*] Removing {len(cuts)} segments "
          f"(~{n_silence} silences, ~{n_filler} fillers): {fmt(cut_total)} cut")
    print(f"[*] Keeping  {len(keeps)} segments: {fmt(keep_total)}")

    if args.dry_run:
        print("\n--- first 20 cuts ---")
        for s, e, r in cuts[:20]:
            print(f"  cut {fmt(s)} -> {fmt(e)}  ({e - s:.2f}s) {r}")
        return

    encoder = detect_encoder() if args.encoder == "auto" else args.encoder
    print(f"[*] Encoder: {encoder}")
    assemble(video, keeps, output, encoder)
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()
