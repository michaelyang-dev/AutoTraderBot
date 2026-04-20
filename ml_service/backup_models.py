#!/usr/bin/env python3
"""
Model Backup & Restore Utilities
=================================
Handles timestamped backups of production model files,
cleanup of old backups (>6 months), and atomic restore.

Usage:
    from backup_models import backup_current, restore_latest, cleanup_old_backups
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR    = Path(__file__).resolve().parent / "data"
BACKUP_DIR  = DATA_DIR / "backups"

MODEL_FILES = ["model.lgb", "model_rf.pkl", "imputer.pkl", "predictions.parquet"]
RETENTION_DAYS = 180  # 6 months


def backup_current(timestamp: str = None, log_fn=print) -> Path:
    """
    Copy current production model files to a timestamped backup directory.

    Args:
        timestamp: Override timestamp string (default: YYYYMMDD_HHMM)
        log_fn: Logging function

    Returns:
        Path to the backup directory created
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    backup_path = BACKUP_DIR / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)

    backed_up = []
    for fname in MODEL_FILES:
        src = DATA_DIR / fname
        if src.exists():
            dst = backup_path / fname
            shutil.copy2(str(src), str(dst))
            size_mb = dst.stat().st_size / 1024 / 1024
            backed_up.append(f"{fname} ({size_mb:.1f}MB)")
        else:
            log_fn(f"  WARNING: {fname} not found — skipping backup")

    log_fn(f"  Backup created: {backup_path}")
    log_fn(f"  Files: {', '.join(backed_up)}")
    return backup_path


def restore_from(backup_path: Path, log_fn=print) -> bool:
    """
    Restore model files from a backup directory to production location.

    Args:
        backup_path: Path to the backup directory
        log_fn: Logging function

    Returns:
        True if restore succeeded
    """
    if not backup_path.exists():
        log_fn(f"  ERROR: Backup path does not exist: {backup_path}")
        return False

    restored = []
    for fname in MODEL_FILES:
        src = backup_path / fname
        if src.exists():
            dst = DATA_DIR / fname
            shutil.copy2(str(src), str(dst))
            restored.append(fname)

    if not restored:
        log_fn(f"  ERROR: No model files found in {backup_path}")
        return False

    log_fn(f"  Restored from {backup_path}: {', '.join(restored)}")
    return True


def restore_latest(log_fn=print) -> bool:
    """Restore from the most recent backup."""
    if not BACKUP_DIR.exists():
        log_fn("  ERROR: No backups directory found")
        return False

    backups = sorted(BACKUP_DIR.iterdir(), reverse=True)
    backups = [b for b in backups if b.is_dir()]

    if not backups:
        log_fn("  ERROR: No backups found")
        return False

    return restore_from(backups[0], log_fn)


def cleanup_old_backups(retention_days: int = RETENTION_DAYS, log_fn=print) -> int:
    """
    Delete backup directories older than retention_days.

    Returns:
        Number of backups deleted
    """
    if not BACKUP_DIR.exists():
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = 0

    for backup_dir in sorted(BACKUP_DIR.iterdir()):
        if not backup_dir.is_dir():
            continue
        try:
            # Parse timestamp from directory name: YYYYMMDD_HHMM
            ts = datetime.strptime(backup_dir.name, "%Y%m%d_%H%M")
            if ts < cutoff:
                shutil.rmtree(str(backup_dir))
                log_fn(f"  Deleted old backup: {backup_dir.name}")
                deleted += 1
        except ValueError:
            continue  # skip non-timestamp directories

    if deleted:
        log_fn(f"  Cleaned up {deleted} backups older than {retention_days} days")
    return deleted


def list_backups(log_fn=print) -> list[Path]:
    """List all available backups, newest first."""
    if not BACKUP_DIR.exists():
        log_fn("  No backups directory")
        return []

    backups = sorted(
        [b for b in BACKUP_DIR.iterdir() if b.is_dir()],
        reverse=True,
    )
    for b in backups:
        files = [f.name for f in b.iterdir() if f.is_file()]
        total_mb = sum(f.stat().st_size for f in b.iterdir() if f.is_file()) / 1024 / 1024
        log_fn(f"  {b.name}  ({total_mb:.1f}MB)  [{', '.join(files)}]")
    return backups


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "backup":
        backup_current()
    elif cmd == "restore":
        if len(sys.argv) > 2:
            restore_from(BACKUP_DIR / sys.argv[2])
        else:
            restore_latest()
    elif cmd == "cleanup":
        cleanup_old_backups()
    elif cmd == "list":
        backups = list_backups()
        if not backups:
            print("No backups found.")
    else:
        print(f"Usage: {sys.argv[0]} [backup|restore [timestamp]|cleanup|list]")
