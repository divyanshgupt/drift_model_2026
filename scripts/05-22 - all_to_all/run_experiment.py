"""
Server script for all-to-all recurrent network experiments.
Mirrors the 05-22 notebook series (1A, 1B, 2A, 2B, 3A, 3B).

Usage:
    # Run all three inh_scales for one condition:
    python run_experiment.py --condition 1A

    # Run a single inh_scale (useful for parallelising across jobs):
    python run_experiment.py --condition 1A --inh_scale 0.3

    # Run every condition sequentially:
    python run_experiment.py --condition all
"""

import matplotlib
matplotlib.use("Agg")  # must come before any other matplotlib import

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "src")
RES_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "results", "05-22 - all-to-all")
sys.path.insert(0, SRC_DIR)

from baseline_network import BaselineNetwork

# ---------------------------------------------------------------------------
# Condition definitions — one entry per notebook
# ---------------------------------------------------------------------------
CONFIGS = {
    "1A": dict(
        label="1A. F to E plastic - co-tuned inh",
        net_kwargs=dict(
            inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
            plasticity_E_to_E="off", plasticity_E_to_I="off",
            plasticity_I_to_E="off", plasticity_I_to_I="off",
            norm=True, set_seed=True, seed=42,
        ),
    ),
    "1B": dict(
        label="1B. F to E plastic - random inh",
        net_kwargs=dict(
            inh_type="random", E_to_E="on", E_to_I="on", I_to_I="on",
            plasticity_E_to_E="off", plasticity_E_to_I="off",
            plasticity_I_to_E="off", plasticity_I_to_I="off",
            norm=True, set_seed=True, seed=42,
        ),
    ),
    "2A": dict(
        label="2A. F to E + F to I plastic - co-tuned inh",
        net_kwargs=dict(
            inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
            plasticity_E_to_E="off", plasticity_E_to_I="off",
            plasticity_I_to_E="off", plasticity_I_to_I="off",
            plasticity_F_to_I="on",
            norm=True, set_seed=True, seed=42,
        ),
    ),
    "2B": dict(
        label="2B. F to E + F to I plastic - random inh",
        net_kwargs=dict(
            inh_type="random", E_to_E="on", E_to_I="on", I_to_I="on",
            plasticity_E_to_E="off", plasticity_E_to_I="off",
            plasticity_I_to_E="off", plasticity_I_to_I="off",
            plasticity_F_to_I="on",
            norm=True, set_seed=True, seed=42,
        ),
    ),
    "3A": dict(
        label="3A. all plastic - co-tuned inh",
        # net_inh_1 config (from notebook): all recurrent connections + all plasticity on
        net_kwargs=dict(
            inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
            plasticity_E_to_E="on", plasticity_E_to_I="on",
            plasticity_I_to_E="on", plasticity_I_to_I="on",
            plasticity_F_to_I="on",
            norm=True, set_seed=True, seed=42,
        ),
    ),
    "3B": dict(
        label="3B. all plastic - random inh",
        net_kwargs=dict(
            inh_type="random", E_to_E="on", E_to_I="on", I_to_I="on",
            plasticity_E_to_E="on", plasticity_E_to_I="on",
            plasticity_I_to_E="on", plasticity_I_to_I="on",
            plasticity_F_to_I="on",
            norm=True, set_seed=True, seed=42,
        ),
    ),
}

INH_SCALES = [0.3, 0.6, 0.9]


def run_condition(condition_key, inh_scales):
    cfg = CONFIGS[condition_key]
    print(f"\n{'='*60}")
    print(f"Condition: {cfg['label']}")
    print(f"inh_scales: {inh_scales}")
    print(f"{'='*60}")

    for scale in inh_scales:
        save_dir = os.path.join(RES_DIR, cfg["label"], f"inh_scale_{scale}")
        print(f"\n--- inh_scale={scale}  →  {save_dir}")

        net = BaselineNetwork(
            **cfg["net_kwargs"],
            inh_scale=scale,
            save_location=save_dir,
        )
        net.run_analysis(save_results=True)
        print(f"Done: {cfg['label']} | inh_scale={scale}")


def main():
    parser = argparse.ArgumentParser(description="Run all-to-all network experiments.")
    parser.add_argument(
        "--condition",
        type=str,
        default="1A",
        choices=list(CONFIGS.keys()) + ["all"],
        help="Which notebook condition to run (1A–3B), or 'all' for every condition.",
    )
    parser.add_argument(
        "--inh_scale",
        type=float,
        default=None,
        help="Single inhibitory scale to run. Omit to run all three (0.3, 0.6, 0.9).",
    )
    args = parser.parse_args()

    scales = [args.inh_scale] if args.inh_scale is not None else INH_SCALES
    conditions = list(CONFIGS.keys()) if args.condition == "all" else [args.condition]

    for cond in conditions:
        run_condition(cond, scales)

    print("\nAll done.")


if __name__ == "__main__":
    main()
