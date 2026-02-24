'''
Author: Chao Deng && chaodeng987@outlook.com
Date: 2026-02-14 16:03:35
LastEditors: Chao Deng && chaodeng987@outlook.com
LastEditTime: 2026-02-14 16:13:35
FilePath: /MultiRM/Scripts/train_treex101.py
Description: 
那只是一场游戏一场梦
 
https://orcid.org/0009-0009-8520-1656
DOI: 10.3390/app15158626
DOI: 10.3390/rs17142354
Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
'''
#!/usr/bin/env python
"""
TreeX training script for sequence length 101.
"""
from treex_utils import parse_args, run_experiment


if __name__ == "__main__":
    # Parse arguments with defaults for sequence length 101
    args = parse_args(
        length=101,
        gpu=[0],
        amp=True,
    )
    run_experiment(args, base_log_dir='logs101')
