#!/usr/bin/env python
"""
一键训练：对所有基金进行半年回测训练 + 7天验证，保存最优参数
用法: python train_models.py

训练完成后，OpenClaw 调用 quant-manager skill 时将自动使用保存的参数进行当日预测，无需每次重算
"""
from __future__ import annotations

import argparse

from training.trainer import run_training, TRAINED_PARAMS_PATH


def main():
    parser = argparse.ArgumentParser(description="量化模型离线训练")
    parser.add_argument("--months", type=int, default=6, help="训练周期（月）")
    parser.add_argument("--test-days", type=int, default=7, help="验证天数")
    parser.add_argument("--max-combos", type=int, default=80, help="每策略最大参数组合数")
    parser.add_argument("-q", "--quiet", action="store_true", help="减少输出")
    args = parser.parse_args()

    result = run_training(
        train_months=args.months,
        test_days=args.test_days,
        max_combos=args.max_combos,
        verbose=not args.quiet,
    )

    print(f"\n训练完成，参数已保存至: {TRAINED_PARAMS_PATH}")
    print("OpenClaw 调用 quant-manager 时将自动使用上述参数进行当日预测。")
    return result


if __name__ == "__main__":
    main()
