#!/usr/bin/env python
"""
TreeX training script for sequence length 1001.
"""
from treex_utils import parse_args, run_experiment


if __name__ == "__main__":
    # Parse arguments with defaults for sequence length 1001
    args = parse_args(
        length=1001,
        gpu=[1],
        amp=True,
    )
    run_experiment(args, base_log_dir='logs-1001')
