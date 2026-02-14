#!/usr/bin/env python
"""
TreeX training script for sequence length 51.
"""
from treex_utils import parse_args, run_experiment


if __name__ == "__main__":
    # Parse arguments with defaults for sequence length 51
    args = parse_args(
        length=51,
        gpu=[0],
        amp=True,
    )
    run_experiment(args, base_log_dir='logs51')
