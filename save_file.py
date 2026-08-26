"""
Export a trained/tested agent's daily buy/sell/hold decisions to a
human-readable CSV, alongside the LLM's reasoning for each day (from its
reflection memory) when available.

Usage:
    python save_file.py --checkpoint data/08_test_checkpoint/agent_1 --out data/09_results/tsla_decisions.csv
"""

import csv
import argparse
from pathlib import Path

from puppy import LLMAgent

DIRECTION_LABEL = {1: "buy", 0: "hold", -1: "sell"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="data/08_test_checkpoint/agent_1",
        help="Path to the agent checkpoint to load (the '<...>/agent_1' directory)",
    )
    parser.add_argument(
        "--out",
        default="data/09_results/tsla_decisions.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    agent = LLMAgent.load_checkpoint(args.checkpoint)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "symbol", "direction", "direction_label", "summary_reason"])
        for cur_date in sorted(agent.portfolio.action_series.keys()):
            direction = agent.portfolio.action_series[cur_date]
            reflection = agent.reflection_result_series_dict.get(cur_date, {}) or {}
            reason = reflection.get("summary_reason", "")
            writer.writerow(
                [
                    cur_date,
                    agent.trading_symbol,
                    direction,
                    DIRECTION_LABEL[direction],
                    reason,
                ]
            )

    print(f"{agent.trading_symbol}: {len(agent.portfolio.action_series)} decisions saved to {out_path}")


if __name__ == "__main__":
    main()
