"""Display TFT training progress bar for a running or completed seed."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from revision_config import REVISION_OUTPUT_DIR

TFT_CHECKPOINT_ROOT = REVISION_OUTPUT_DIR / "checkpoints" / "tft"
DEFAULT_EPOCH_MINUTES = 16.0  # empirical from seed 20250113 epoch 0
MAX_EPOCHS = 40


def bar(fraction: float, width: int = 40) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(width * fraction))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def completed_epochs(ckpt_dir: Path) -> int:
    best = list(ckpt_dir.glob("best-epoch=*.ckpt"))
    if not best:
        return 0
    epochs = []
    for path in best:
        m = re.search(r"best-epoch=(\d+)", path.name)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) + 1 if epochs else 0


def read_progress(ckpt_dir: Path) -> dict | None:
    path = ckpt_dir / "progress.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_epoch_fraction(ckpt_dir: Path, resume_time: datetime | None) -> tuple[int, float, str, bool]:
    progress = read_progress(ckpt_dir)
    if progress:
        epoch = int(progress.get("epoch", 0))
        return epoch, float(progress.get("epoch_fraction", 0.0)), str(progress.get("phase", "unknown")), True

    done = completed_epochs(ckpt_dir)
    current_epoch = done
    if resume_time is None:
        return current_epoch, 0.0, "waiting (no live progress.json)", False

    elapsed_min = (datetime.now() - resume_time).total_seconds() / 60.0
    if elapsed_min <= DEFAULT_EPOCH_MINUTES:
        frac = elapsed_min / DEFAULT_EPOCH_MINUTES * 0.85
        phase = "train (time-estimated)"
    elif elapsed_min <= DEFAULT_EPOCH_MINUTES * 1.2:
        frac = 0.85 + (elapsed_min - DEFAULT_EPOCH_MINUTES) / (DEFAULT_EPOCH_MINUTES * 0.2) * 0.15
        phase = "validation (time-estimated)"
    else:
        frac = min(0.98, 0.85 + (elapsed_min - DEFAULT_EPOCH_MINUTES) / DEFAULT_EPOCH_MINUTES * 0.1)
        phase = f"still running ({elapsed_min:.0f} min, past typical epoch length)"
    return current_epoch, frac, phase, False


def process_alive() -> tuple[bool, str]:
    try:
        import subprocess

        out = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"], text=True)
        for line in out.splitlines():
            if "14_tft_stable" in line or "python.exe" in line.lower():
                pass
        # check CPU delta on likely training PID via wmic
        wmic = subprocess.check_output(
            ["wmic", "process", "where", "CommandLine like '%14_tft_stable%train%'", "get", "ProcessId,WorkingSetSize", "/format:csv"],
            text=True,
            errors="replace",
        )
        lines = [ln.strip() for ln in wmic.splitlines() if ln.strip() and "ProcessId" not in ln and ln.strip().replace(",", "")]
        if not lines:
            return False, "no TFT train process found"
        pid = lines[0].split(",")[-1].strip()
        import psutil  # may not exist
    except Exception:
        pid = None

    try:
        import psutil

        matches = [
            p
            for p in psutil.process_iter(["pid", "name", "cmdline"])
            if p.info.get("name") == "python.exe"
            and p.info.get("cmdline")
            and any("14_tft_stable" in str(c) and "train" in str(c) for c in p.info["cmdline"])
        ]
        if not matches:
            return False, "no TFT train process found"
        p = matches[0]
        t0 = p.cpu_times().user + p.cpu_times().system
        time.sleep(3)
        t1 = p.cpu_times().user + p.cpu_times().system
        delta = t1 - t0
        ws = p.memory_info().rss / 1024**2
        if delta > 1.0:
            return True, f"PID {p.pid} active (CPU +{delta:.1f}s/3s, RAM {ws:.0f} MB)"
        return False, f"PID {p.pid} idle (CPU +{delta:.2f}s/3s) — may be stuck"
    except ImportError:
        return True, "process check inconclusive (install psutil for live CPU probe)"


def show_progress(seed: int, once: bool = True, refresh: float = 10.0) -> None:
    ckpt_dir = TFT_CHECKPOINT_ROOT / str(seed)
    log_path = REVISION_OUTPUT_DIR / "logs" / f"tft_{seed}_training.log"

    resume_time = None
    if log_path.exists():
        for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if "Entering trainer.fit resume=" in line:
                ts = line.split(" [INFO] ")[0]
                for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
                    try:
                        resume_time = datetime.strptime(ts, fmt)
                        break
                    except ValueError:
                        continue
                if resume_time is not None:
                    break

    while True:
        done_epochs = completed_epochs(ckpt_dir)
        current_epoch, epoch_frac, phase, live = estimate_epoch_fraction(ckpt_dir, resume_time)
        overall = min(1.0, (done_epochs + epoch_frac) / MAX_EPOCHS)
        marker = ckpt_dir / "training_complete.marker"
        complete = marker.exists()
        alive, alive_msg = process_alive()

        print(f"\nTFT seed {seed} training progress")
        print(f"Process: {alive_msg}")
        print(f"Completed epochs (checkpoints): {done_epochs}")
        print(f"Current epoch: {current_epoch}  phase: {phase}" + (" [live batch tracking]" if live else " [time estimate only]"))
        print(f"Epoch progress: {bar(epoch_frac)} {epoch_frac * 100:5.1f}%")
        print(f"Overall (max {MAX_EPOCHS} epochs): {bar(overall)} {overall * 100:5.1f}%")
        if resume_time:
            elapsed = datetime.now() - resume_time
            print(f"Elapsed this run: {str(elapsed).split('.')[0]}  (~{DEFAULT_EPOCH_MINUTES:.0f} min typical per epoch)")
        if complete:
            print("Status: TRAINING COMPLETE")
        elif not alive:
            print("Status: STOPPED OR STUCK — no active compute detected")
        elif not live and done_epochs <= current_epoch:
            print("Status: RUNNING — waiting for next checkpoint (progress bar is approximate)")
        else:
            print("Status: RUNNING")

        best = sorted(ckpt_dir.glob("best-epoch=*.ckpt"))
        if best:
            print(f"Best checkpoint: {best[-1].name}")

        if once or complete:
            break
        time.sleep(refresh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show TFT training progress bar.")
    parser.add_argument("--seed", type=int, default=20250113)
    parser.add_argument("--watch", action="store_true", help="Refresh every 10 seconds.")
    args = parser.parse_args()
    show_progress(args.seed, once=not args.watch)


if __name__ == "__main__":
    main()
