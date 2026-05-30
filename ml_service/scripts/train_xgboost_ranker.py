#!/usr/bin/env python3
"""
Train XGBoost LambdaRank Stock Ranker
=====================================
Trains from scratch using WRDS universe data. Walk-forward validated.

Usage:
    cd ml_service && OMP_NUM_THREADS=1 python3 scripts/train_xgboost_ranker.py
"""
import logging
import sys
import os

# Prevent XGBoost/OpenMP hangs on Mac
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

from strategies.xgboost_ranker import XGBoostRanker

UNIVERSE_PATH = "data/wrds/complete_sp1500_universe.pkl"
MODEL_PATH = "data/xgboost_ranker_v1.pkl"


def main():
    log.info("=" * 60)
    log.info("  XGBOOST RANKER TRAINING")
    log.info("=" * 60)

    ranker = XGBoostRanker()

    # Train with walk-forward validation
    results = ranker.train_from_universe(UNIVERSE_PATH, save_path=MODEL_PATH)

    # Print results
    log.info("\n" + "=" * 60)
    log.info("WALK-FORWARD RESULTS")
    log.info("=" * 60)

    for r in results:
        log.info(f"  Train {r['train_period']} → Val {r['val_year']}: "
                 f"NDCG@8 = {r['ndcg_at_8']:.4f} "
                 f"(n_train={r['n_train']}, n_val={r['n_val']}, "
                 f"best_iter={r['best_iter']})")

    avg_ndcg = sum(r["ndcg_at_8"] for r in results) / len(results) if results else 0
    log.info(f"\nAverage NDCG@8: {avg_ndcg:.4f}")
    log.info(f"Model saved to: {MODEL_PATH}")

    # Baseline: random ranking NDCG@8 ≈ 0.85 for 10-decile labels
    # Good model should be > 0.87
    if avg_ndcg > 0.87:
        log.info("PASS — model beats random baseline")
    else:
        log.warning("WEAK — model barely beats random. Consider not deploying.")


if __name__ == "__main__":
    main()
