#!/usr/bin/env python3
"""Run Muaalem phoneme inference over the full tarteel-ai/tlog dataset.

Streams the dataset one parquet shard at a time (411 shards in the
`clean` split): downloads a shard, runs inference on every clip in it,
writes the results, deletes the shard's audio from local disk, then
moves on to the next one. The full dataset is ~1000 hours of audio, so
it is never fully present on local disk at once -- only one shard
(a few hundred MB) is.

Setup (once, before running):
    pip install "datasets[audio]>4.0.0" soundfile soxr huggingface_hub quran-muaalem
    export HF_TOKEN=...  # needs read access to the gated tarteel-ai/tlog dataset

GPU note: defaults to float16, not Muaalem's own bfloat16 default. A T4
(compute capability 7.5) has no native bf16 tensor cores -- fp16 is the
fast, well-supported choice there. Override with --dtype if you're on
newer hardware.

Storing results in Google Drive: this script has no Google Drive API
code of its own. Instead it optionally shells out to `rclone` (the
standard way to talk to Drive from a plain Linux box like a Lightning
Studio -- Colab-style `drive.mount()` doesn't exist there). One-time
setup: `rclone config` and create a remote (e.g. named `gdrive`)
pointing at your Google Drive. Then pass
`--rclone-remote gdrive:tlog_inference_results` and each shard's
result file is copied there right after it's written (and the local
copy deleted, unless --keep-local-results is passed). Without
--rclone-remote, results just accumulate under --output-dir and you're
responsible for getting them onto Drive yourself.

Resuming: a shard counts as "done" if a
`<split>-NNNNN-of-NNNNN.results.jsonl` file for it already exists
locally or (if --rclone-remote is set) at the remote -- so reruns
(including on a fresh, ephemeral machine, as long as --rclone-remote
points at the same Drive folder) automatically skip finished shards.
Interrupting mid-shard is safe: results are written to a `.tmp` file
and only renamed to the final name once the whole shard is done, so a
half-finished shard is simply redone from scratch next run.

Batching: clips are only ever batched together with other clips of the
*same ayah* (never mixed across ayat). This isn't just an optimization
-- Muaalem tokenizes reference phonetic scripts with padding="longest",
and quran_muaalem.decode.multilevel_greedy_decode doesn't account for
that padding when aligning sifat levels back to the predicted phonemes,
so batching clips with different-length references currently raises
`IndexError: The shape of the mask ... does not match ...`. Since two
clips of the same ayah always have the exact same reference (it only
depends on surah/ayah + the fixed Hafs moshaf below, never on the
audio), grouping by ayah sidesteps the bug entirely while still
batching whenever a shard happens to have multiple clips of one ayah.

Example usage:
    # Smoke test: just the first 2 shards, small batch size
    python scripts/run_tlog_phoneme_inference.py --limit-files 2 --batch-size 4

    # Full run, syncing each finished shard to Google Drive via rclone
    python scripts/run_tlog_phoneme_inference.py \\
        --rclone-remote gdrive:tlog_inference_results

    # Resume an interrupted run -- just run the same command again
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import soxr
import torch
from datasets import Audio, Dataset
from huggingface_hub import HfApi, hf_hub_download
from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

from quran_muaalem import Muaalem

logger = logging.getLogger("tlog_inference")

TARGET_SR = 16000

# NOTE: assumes every clip in tlog is Hafs recitation. Confirmed by
# manual listening during exploration (see notebooks/explore_tlog.ipynb);
# revisit if the dataset schema turns out to carry a riwayah/qiraat tag.
DEFAULT_MOSHAF = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=2,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=2,
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def get_sura_aya_key(path: str | None) -> str | None:
    """Same parsing as load_tarteel_log.ipynb -- first two `_`-separated tokens."""
    if not path:
        return None
    filename = os.path.basename(path)
    parts = filename.split("_")
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return None


def load_wave_16k(audio_bytes: bytes):
    wave, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:
        wave = soxr.resample(wave, sr, TARGET_SR)
    return wave


def build_reference(surah: int, ayah: int, moshaf: MoshafAttributes = DEFAULT_MOSHAF):
    uthmani = Aya(surah, ayah).get().uthmani
    phonetic = quran_phonetizer(uthmani, moshaf, remove_spaces=True)
    return uthmani, phonetic


def to_serializable(value):
    """Recursively turn dataclasses/tensors from a MuaalemOutput into plain JSON types."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_serializable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def rclone_list_existing(remote: str) -> set[str]:
    """Filenames already present at an rclone remote path (empty if it doesn't exist yet)."""
    try:
        out = subprocess.run(
            ["rclone", "lsf", remote],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Could not list rclone remote %s (%s); assuming empty", remote, e)
        return set()
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def rclone_copy(local_path: Path, remote: str) -> None:
    subprocess.run(["rclone", "copy", str(local_path), remote], check=True)


def list_done_shard_indices(results_dir: Path, rclone_remote: str | None) -> set[int]:
    names = set()
    if results_dir.exists():
        names |= {p.name for p in results_dir.glob("*.results.jsonl")}
    if rclone_remote:
        names |= rclone_list_existing(rclone_remote)

    indices = set()
    for name in names:
        # e.g. "clean-00007-of-00411.results.jsonl" -> 7
        parts = name.split("-")
        if len(parts) >= 2:
            try:
                indices.add(int(parts[1]))
            except ValueError:
                continue
    return indices


def process_shard(
    file_index: int,
    num_files: int,
    split: str,
    dataset_repo: str,
    hf_token: str,
    muaalem: Muaalem,
    batch_size: int,
    min_duration: float | None,
    max_duration: float | None,
    tmp_root: Path,
) -> list[dict]:
    filename = f"{split}-{file_index:05d}-of-{num_files:05d}.parquet"
    tmp_dir = Path(tempfile.mkdtemp(prefix="tlog_shard_", dir=str(tmp_root)))
    results: list[dict] = []

    try:
        local_path = hf_hub_download(
            repo_id=dataset_repo,
            filename=f"data/{filename}",
            repo_type="dataset",
            token=hf_token,
            cache_dir=str(tmp_dir),
        )
        ds = Dataset.from_parquet(local_path).cast_column("audio", Audio(decode=False))
        audio_col = ds["audio"]  # list[dict] -- avoids repeated per-row arrow lookups below

        groups: dict[str | None, list[int]] = {}
        for idx, audio in enumerate(audio_col):
            key = get_sura_aya_key(audio["path"])
            groups.setdefault(key, []).append(idx)

        ref_cache: dict[str, tuple[str, object] | None] = {}

        for key, indices in groups.items():
            if key is None:
                for idx in indices:
                    results.append(
                        {
                            "shard_file": filename,
                            "row_index": idx,
                            "audio_path": audio_col[idx]["path"],
                            "status": "skipped",
                            "reason": "could_not_parse_surah_ayah",
                        }
                    )
                continue

            surah, ayah = (int(x) for x in key.split("_"))

            if key not in ref_cache:
                try:
                    ref_cache[key] = build_reference(surah, ayah)
                except Exception as e:
                    logger.warning("Failed to build reference for ayah %s: %s", key, e)
                    ref_cache[key] = None
            ref_entry = ref_cache[key]

            if ref_entry is None:
                for idx in indices:
                    results.append(
                        {
                            "shard_file": filename,
                            "row_index": idx,
                            "audio_path": audio_col[idx]["path"],
                            "surah": surah,
                            "ayah": ayah,
                            "status": "skipped",
                            "reason": "reference_build_failed",
                        }
                    )
                continue

            uthmani_text, phonetic_ref = ref_entry

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                waves, row_meta = [], []

                for idx in batch_indices:
                    audio = audio_col[idx]
                    path = audio["path"]
                    base_row = {
                        "shard_file": filename,
                        "row_index": idx,
                        "audio_path": path,
                        "surah": surah,
                        "ayah": ayah,
                    }

                    if not audio.get("bytes"):
                        results.append({**base_row, "status": "skipped", "reason": "no_audio_bytes"})
                        continue

                    try:
                        with io.BytesIO(audio["bytes"]) as f:
                            duration = sf.info(f).duration
                    except Exception as e:
                        results.append(
                            {**base_row, "status": "skipped", "reason": f"sf_info_failed: {e}"}
                        )
                        continue

                    if min_duration is not None and duration < min_duration:
                        results.append(
                            {
                                **base_row,
                                "duration_sec": duration,
                                "status": "skipped",
                                "reason": "too_short",
                            }
                        )
                        continue
                    if max_duration is not None and duration > max_duration:
                        results.append(
                            {
                                **base_row,
                                "duration_sec": duration,
                                "status": "skipped",
                                "reason": "too_long",
                            }
                        )
                        continue

                    try:
                        wave = load_wave_16k(audio["bytes"])
                    except Exception as e:
                        results.append(
                            {
                                **base_row,
                                "duration_sec": duration,
                                "status": "skipped",
                                "reason": f"decode_failed: {e}",
                            }
                        )
                        continue

                    waves.append(wave)
                    row_meta.append((base_row, duration))

                if not waves:
                    continue

                try:
                    outs = muaalem(
                        waves, [phonetic_ref] * len(waves), sampling_rate=TARGET_SR
                    )
                except Exception as e:
                    logger.warning(
                        "Inference failed for a batch of ayah %s in %s: %s", key, filename, e
                    )
                    for base_row, duration in row_meta:
                        results.append(
                            {
                                **base_row,
                                "duration_sec": duration,
                                "status": "error",
                                "reason": f"inference_failed: {e}",
                            }
                        )
                    continue

                for (base_row, duration), out in zip(row_meta, outs):
                    results.append(
                        {
                            **base_row,
                            "duration_sec": round(duration, 3),
                            "status": "ok",
                            "reference_uthmani": uthmani_text,
                            "reference_phonemes": phonetic_ref.phonemes,
                            "predicted_phonemes": out.phonemes.text,
                            "predicted_phoneme_probs": to_serializable(out.phonemes.probs),
                            "sifat": to_serializable(out.sifat),
                        }
                    )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-repo", default="tarteel-ai/tlog")
    parser.add_argument("--split", default="clean")
    parser.add_argument("--hf-token", default=None, help="Defaults to $HF_TOKEN env var")
    parser.add_argument("--model", default="obadx/muaalem-model-v3_2")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=sorted(DTYPE_MAP),
        help="float16 is the safe default for a T4; T4 has poor bfloat16 support.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Max clips per inference call. Only ever batches clips of the same ayah -- see "
        "the module docstring for why.",
    )
    parser.add_argument("--min-duration", type=float, default=None, help="Seconds; clips shorter than this are skipped.")
    parser.add_argument("--max-duration", type=float, default=None, help="Seconds; clips longer than this are skipped.")
    parser.add_argument("--output-dir", default="./tlog_inference_output")
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Where downloaded parquet shards live before being deleted; "
        "defaults to <output-dir>/tlog_shards_tmp.",
    )
    parser.add_argument(
        "--rclone-remote",
        default=None,
        help="e.g. gdrive:tlog_inference_results -- if set, each finished shard's results are "
        "synced there with `rclone copy` right after being written.",
    )
    parser.add_argument(
        "--keep-local-results",
        action="store_true",
        help="Keep local result files after syncing to --rclone-remote (default: delete them "
        "once synced, to save disk).",
    )
    parser.add_argument("--start-file", type=int, default=0)
    parser.add_argument(
        "--end-file", type=int, default=None, help="Exclusive; defaults to all files in the split."
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Process at most this many *new* shards this run -- useful for a smoke test.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("No HF token: pass --hf-token or set $HF_TOKEN")

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(args.tmp_dir) if args.tmp_dir else output_dir / "tlog_shards_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=hf_token)
    repo_files = api.list_repo_files(args.dataset_repo, repo_type="dataset")
    split_files = sorted(
        f
        for f in repo_files
        if f.startswith("data/") and Path(f).name.startswith(f"{args.split}-")
    )
    num_files = len(split_files)
    logger.info("Found %d files in split '%s'", num_files, args.split)

    end_file = args.end_file if args.end_file is not None else num_files
    file_indices = list(range(args.start_file, end_file))

    done = list_done_shard_indices(results_dir, args.rclone_remote)
    remaining = [i for i in file_indices if i not in done]
    logger.info(
        "%d/%d requested shards already done, %d remaining",
        len(file_indices) - len(remaining),
        len(file_indices),
        len(remaining),
    )

    logger.info("Loading Muaalem model %s on %s (%s)...", args.model, args.device, args.dtype)
    muaalem = Muaalem(
        model_name_or_path=args.model, device=args.device, dtype=DTYPE_MAP[args.dtype]
    )

    processed_this_run = 0
    for file_index in remaining:
        if args.limit_files is not None and processed_this_run >= args.limit_files:
            logger.info("Reached --limit-files=%d, stopping", args.limit_files)
            break

        logger.info("Processing shard %d/%d", file_index, num_files)
        try:
            results = process_shard(
                file_index=file_index,
                num_files=num_files,
                split=args.split,
                dataset_repo=args.dataset_repo,
                hf_token=hf_token,
                muaalem=muaalem,
                batch_size=args.batch_size,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                tmp_root=tmp_root,
            )
        except Exception:
            logger.exception("Shard %d failed entirely, will retry next run", file_index)
            continue

        filename = f"{args.split}-{file_index:05d}-of-{num_files:05d}.results.jsonl"
        tmp_result_path = results_dir / f"{filename}.tmp"
        final_result_path = results_dir / filename
        with open(tmp_result_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_result_path.rename(final_result_path)
        logger.info("Wrote %d rows -> %s", len(results), final_result_path)

        if args.rclone_remote:
            try:
                rclone_copy(final_result_path, args.rclone_remote)
                logger.info("Synced %s -> %s", final_result_path.name, args.rclone_remote)
                if not args.keep_local_results:
                    final_result_path.unlink()
            except subprocess.CalledProcessError as e:
                logger.error(
                    "rclone sync failed for %s: %s -- keeping local copy", filename, e
                )

        processed_this_run += 1

    shutil.rmtree(tmp_root, ignore_errors=True)
    logger.info("Done. Processed %d new shard(s) this run.", processed_this_run)


if __name__ == "__main__":
    main()
