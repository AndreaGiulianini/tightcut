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
    ./tightcut.py input.mov --dry-run                   # see what would be cut
    ./tightcut.py input.mov --model large-v3            # max accuracy (slower)
    ./tightcut.py input.mov --backend faster-whisper    # force CPU backend
    ./tightcut.py input.mov --encode-mode full          # re-encode everything (most compatible)
"""

from __future__ import annotations

import argparse
import bisect
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

HESITATION_FILLERS = {
    "ehm", "ehmm", "ehmmm", "uhm", "uhmm", "mh", "mhm", "mmm", "mmmm", "mmh",
    "hmm", "eh", "ehh", "ah", "ahh", "uh", "uhh", "eee", "ee", "umm",
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

ENCODER_ARGS = {
    "h264_videotoolbox": ["-b:v", "12M"],
    "h264_nvenc": ["-cq", "22", "-preset", "p5"],
    "libx264": ["-crf", "20", "-preset", "veryfast"],
}


@dataclass
class Word:
    text: str
    start: float
    end: float


_QUIET = False


def info(msg: str) -> None:
    """Print an info line unless --quiet is in effect."""
    if not _QUIET:
        print(msg)


def warn(msg: str) -> None:
    """Print a warning to stderr (always shown, even under --quiet)."""
    print(msg, file=sys.stderr)


def run_ff(cmd: list[str], what: str, *, stream: bool = False) -> None:
    """Run an ffmpeg subprocess; on failure raise SystemExit with the ffmpeg tail.

    `stream=True` tees stderr to the user's terminal in real time (so long
    encodes show progress) while still buffering the last lines for a useful
    error message. Always silent under --quiet.
    """
    if stream and not _QUIET:
        tail: list[str] = []
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)
        assert proc.stderr is not None
        for line in proc.stderr:
            sys.stderr.write(line)
            tail.append(line)
            if len(tail) > 40:
                tail = tail[-40:]
        proc.wait()
        rc, last = proc.returncode, "".join(tail[-20:])
    else:
        completed = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        rc = completed.returncode
        last = "\n".join(completed.stderr.splitlines()[-20:])
    if rc != 0:
        raise SystemExit(f"[!] {what} failed (exit {rc}):\n{last}")


def check_ff_output(cmd: list[str], what: str) -> str:
    """Like subprocess.check_output but raises SystemExit with the ffmpeg tail."""
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise SystemExit(f"[!] {what} failed (exit {completed.returncode}):\n{tail}")
    return completed.stdout


def probe_duration(path: Path) -> float:
    out = check_ff_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], f"ffprobe (duration of {path.name})")
    return float(out.strip())


def probe_keyframes(path: Path) -> list[float]:
    """Sorted list of video keyframe pts_time values."""
    out = check_ff_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
        "-of", "csv=p=0", str(path),
    ], f"ffprobe (keyframes of {path.name})")
    # ffprobe CSV may emit a trailing field separator; take the first column.
    return sorted(
        float(line.split(",", 1)[0])
        for line in out.splitlines()
        if line.strip() and line.split(",", 1)[0].strip()
    )


def load_or_probe_keyframes(video: Path, no_cache: bool) -> list[float]:
    """probe_keyframes with a sidecar JSON cache invalidated on mtime/size."""
    cache_path = video.with_name(video.name + ".keyframes.json")
    if cache_path.exists() and not no_cache:
        try:
            data = json.loads(cache_path.read_text())
            stat = video.stat()
            if (
                data.get("version") == 1
                and data.get("mtime") == stat.st_mtime
                and data.get("size") == stat.st_size
            ):
                info(f"[*] Reusing cached keyframes: {cache_path.name}")
                return data["keyframes"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass  # fall through and re-probe
    info("[*] Probing keyframes...")
    keyframes = probe_keyframes(video)
    stat = video.stat()
    cache_path.write_text(json.dumps({
        "version": 1, "mtime": stat.st_mtime, "size": stat.st_size, "keyframes": keyframes,
    }))
    info(f"[*] Cached keyframes -> {cache_path.name}")
    return keyframes


def split_keep_smart(
    s: float, e: float, keyframes: list[float], min_head: float = 0.05,
) -> list[tuple[float, float, str]]:
    """Split one keep into (encode-head?) + (copy-body?) sub-segments.

    Stream-copy needs to start on a keyframe, so we re-encode any leading
    fragment from `s` up to the first keyframe inside the range, then copy
    the rest. If `s` is already within `min_head` of the next keyframe, snap
    forward (avoids generating a sub-frame head).
    """
    i = bisect.bisect_right(keyframes, s)
    if i < len(keyframes) and 0 < keyframes[i] - s < min_head:
        s = keyframes[i]
        i += 1
    on_keyframe = i > 0 and abs(keyframes[i - 1] - s) < 1e-3
    if on_keyframe:
        body_start = s
    else:
        next_kf = keyframes[i] if i < len(keyframes) else e
        if next_kf >= e:
            return [(s, e, "encode")]
        body_start = next_kf
    pieces: list[tuple[float, float, str]] = []
    if body_start > s:
        pieces.append((s, body_start, "encode"))
    if e > body_start:
        pieces.append((body_start, e, "copy"))
    return pieces


def extract_audio(video: Path, wav: Path) -> None:
    info(f"[*] Extracting audio to {wav.name} (16 kHz mono PCM)...")
    run_ff([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav),
    ], "audio extraction")


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
    if shutil.which("nvidia-smi") is not None:
        return "h264_nvenc"
    return "libx264"


def transcribe(wav: Path, language: str, model_size: str, backend: str) -> list[Word]:
    if backend == "auto":
        backend = detect_backend()
    info(f"[*] Backend: {backend}  |  Model: {model_size}  |  Language: {language}")
    if backend == "mlx":
        return _transcribe_mlx(wav, language, model_size)
    return _transcribe_faster_whisper(wav, language, model_size)


def load_or_transcribe(
    video: Path, language: str, model: str, backend: str, no_cache: bool
) -> list[Word]:
    cache_path = video.with_name(video.name + ".whisper.json")
    if cache_path.exists() and not no_cache:
        info(f"[*] Reusing cached transcription: {cache_path.name} (use --no-cache to redo)")
        return [Word(**w) for w in json.loads(cache_path.read_text())]
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        extract_audio(video, wav)
        words = transcribe(wav, language, model, backend)
    cache_path.write_text(json.dumps([asdict(w) for w in words], ensure_ascii=False))
    info(f"[*] Cached transcription -> {cache_path.name}")
    return words


def _transcribe_mlx(wav: Path, language: str, model_size: str) -> list[Word]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise SystemExit(
            "mlx-whisper not installed. It only runs on Apple Silicon. "
            "Use --backend faster-whisper to fall back, or run on a Mac."
        ) from exc
    repo = MLX_MODEL_MAP.get(model_size, model_size)
    info(f"[*] Loading mlx-whisper '{repo}' (first run downloads it from HF)...")
    info("[*] Transcribing on Apple GPU/Neural Engine...")
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
    info(f"[*] Loading faster-whisper '{model_size}' (first run downloads it)...")
    # int8 CPU is the most portable path; on CUDA you can edit this to ("cuda", "float16").
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    info("[*] Transcribing on CPU...")
    segments, meta = model.transcribe(
        str(wav),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 250},
        beam_size=5,
        initial_prompt=VERBATIM_PROMPT_IT if language == "it" else None,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    pbar = tqdm(total=round(meta.duration, 2), unit="s", disable=_QUIET,
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
    # strips hyphen, em dash and en dash on purpose
    return s.strip(" .,?!\"'…-—–:;()").lower()  # noqa: RUF001


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
    mode: str,
    no_cache: bool,
) -> None:
    if not keeps:
        raise SystemExit("No segments to keep -- everything looked like silence/fillers.")
    if mode == "full":
        _assemble_full(video, keeps, output, encoder)
        return
    keyframes = load_or_probe_keyframes(video, no_cache)
    gop = (keyframes[-1] - keyframes[0]) / max(1, len(keyframes) - 1) if len(keyframes) >= 2 else 0.0
    info(f"[*] Found {len(keyframes)} keyframes (~{gop:.2f}s GOP)")
    _assemble_smart(video, keeps, output, encoder, keyframes)


def _assemble_full(
    video: Path, keeps: list[tuple[float, float]], output: Path, encoder: str,
) -> None:
    info(f"[*] Assembling {len(keeps)} kept segments -> {output.name}")
    parts: list[str] = []
    for i, (s, e) in enumerate(keeps):
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
    chain = "".join(f"[v{i}][a{i}]" for i in range(len(keeps)))
    parts.append(f"{chain}concat=n={len(keeps)}:v=1:a=1[outv][outa]")

    with tempfile.TemporaryDirectory() as td:
        script_path = Path(td) / "filter.txt"
        script_path.write_text(";\n".join(parts))
        cmd = [
            "ffmpeg", "-y", "-stats", "-loglevel", "error",
            "-i", str(video),
            "-filter_complex_script", str(script_path),
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", encoder, *ENCODER_ARGS[encoder], "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ]
        run_ff(cmd, "ffmpeg full-encode", stream=True)


def _run_video_segment(
    video: Path, start: float, end: float, mode: str, encoder: str, out_path: Path,
) -> None:
    """Encode or stream-copy a single VIDEO-ONLY sub-segment.

    Audio is dropped here and rebuilt precisely in one pass elsewhere -- mixing
    per-fragment audio re-encodes is unsafe (AAC priming offsets accumulate at
    each fragment boundary) and `-c:a copy` over `-ss` skews to the demuxer's
    keyframe-aligned position rather than `start`.
    """
    dur = f"{end - start:.3f}"
    if mode == "encode":
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(video), "-t", dur,
            "-an",
            "-c:v", encoder, *ENCODER_ARGS[encoder], "-pix_fmt", "yuv420p",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(video), "-t", dur,
            "-an",
            "-c:v", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]
    run_ff(cmd, f"ffmpeg segment {mode} @ {start:.2f}s")


def _build_audio_track(
    video: Path, keeps: list[tuple[float, float]], out_audio: Path,
) -> None:
    """Render the kept audio ranges as one continuous AAC track."""
    parts: list[str] = []
    for i, (s, e) in enumerate(keeps):
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
    chain = "".join(f"[a{i}]" for i in range(len(keeps)))
    parts.append(f"{chain}concat=n={len(keeps)}:v=0:a=1[outa]")
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "afilter.txt"
        script.write_text(";\n".join(parts))
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video),
            "-filter_complex_script", str(script),
            "-map", "[outa]", "-vn",
            "-c:a", "aac", "-b:a", "192k",
            str(out_audio),
        ]
        run_ff(cmd, "ffmpeg audio build")


def _concat_copy(parts: list[Path], output: Path, extra_args: list[str] | None = None) -> None:
    with tempfile.TemporaryDirectory() as td:
        list_path = Path(td) / "concat.txt"
        list_path.write_text("\n".join(f"file '{p}'" for p in parts) + "\n")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy", *(extra_args or []), str(output),
        ]
        run_ff(cmd, "ffmpeg concat")


def _mux_av(video: Path, audio: Path, output: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c", "copy", "-shortest",
        "-movflags", "+faststart", str(output),
    ]
    run_ff(cmd, "ffmpeg mux")


def _assemble_smart(
    video: Path,
    keeps: list[tuple[float, float]],
    output: Path,
    encoder: str,
    keyframes: list[float],
) -> None:
    pieces: list[tuple[float, float, str]] = []
    for s, e in keeps:
        pieces.extend(split_keep_smart(s, e, keyframes))
    enc_dur = sum(e - s for s, e, m in pieces if m == "encode")
    cp_dur = sum(e - s for s, e, m in pieces if m == "copy")
    n_enc = sum(1 for _, _, m in pieces if m == "encode")
    n_cp = len(pieces) - n_enc
    info(f"[*] Smart-cut: re-encode {n_enc} video fragments ({fmt(enc_dur)}), "
         f"stream-copy {n_cp} ({fmt(cp_dur)})")
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        parts: list[Path] = []
        for i, (s, e, mode) in enumerate(
            tqdm(pieces, desc="video", unit="seg", smoothing=0.1, disable=_QUIET)
        ):
            part = out_dir / f"v_{i:05d}.mov"
            _run_video_segment(video, s, e, mode, encoder, part)
            parts.append(part)
        video_concat = out_dir / "video.mov"
        info(f"[*] Concatenating {len(parts)} video fragments")
        _concat_copy(parts, video_concat)
        audio_path = out_dir / "audio.m4a"
        info(f"[*] Building audio track ({fmt(sum(e - s for s, e in keeps))})")
        _build_audio_track(video, keeps, audio_path)
        info(f"[*] Muxing -> {output.name}")
        _mux_av(video_concat, audio_path, output)


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
    p.add_argument("--language", default="it",
                   help="audio language code for whisper [default: it]")
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
                   help="cut silences only, keep all words (overrides --aggressive)")
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "h264_videotoolbox", "libx264", "h264_nvenc"],
                   help="auto picks h264_videotoolbox on macOS, libx264 elsewhere")
    p.add_argument("--encode-mode", default="smart",
                   choices=["smart", "full"],
                   help="smart: re-encode only sub-keyframe heads, stream-copy the rest [default]; "
                        "full: re-encode everything via filter_complex (slower, most compatible)")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore cached transcription and keyframes; redo from scratch")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be cut, skip the encode")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress info messages and progress bars (errors and [OK] still shown)")
    args = p.parse_args()

    global _QUIET
    _QUIET = args.quiet

    video = args.input.resolve()
    if not video.exists():
        sys.exit(f"input not found: {video}")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not on PATH -- run: brew install ffmpeg")

    output = args.output or video.with_name(video.stem + ".cut.mov")
    duration = probe_duration(video)
    info(f"[*] Input: {video.name} ({fmt(duration)})")

    fillers: set[str] = set()
    if not args.no_fillers:
        fillers |= HESITATION_FILLERS
        if args.aggressive:
            fillers |= DISCOURSE_FILLERS

    words = load_or_transcribe(video, args.language, args.model, args.backend, args.no_cache)

    cuts = build_cuts(words, duration, fillers, args.max_silence, args.pad)
    keeps = invert(cuts, duration)
    cut_total = sum(e - s for s, e, _ in cuts)
    keep_total = sum(e - s for s, e in keeps)
    n_silence = sum(1 for c in cuts if c[2] in ("silence", "mixed"))
    n_filler = len(cuts) - n_silence
    info(f"[*] Removing {len(cuts)} segments "
         f"(~{n_silence} silences, ~{n_filler} fillers): {fmt(cut_total)} cut")
    info(f"[*] Keeping  {len(keeps)} segments: {fmt(keep_total)}")

    if args.dry_run:
        # --dry-run output is the user-facing payload of the command; show it
        # even under --quiet, since suppressing it would defeat the purpose.
        print("\n--- first 20 cuts ---")
        for s, e, r in cuts[:20]:
            print(f"  cut {fmt(s)} -> {fmt(e)}  ({e - s:.2f}s) {r}")
        return

    encoder = detect_encoder() if args.encoder == "auto" else args.encoder
    info(f"[*] Encoder: {encoder}  |  Mode: {args.encode_mode}")
    assemble(video, keeps, output, encoder, args.encode_mode, args.no_cache)
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()
