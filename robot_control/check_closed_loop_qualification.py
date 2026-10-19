"""Read-only qualification check invoked before native production ROS initialization."""

import argparse

from robot_control.closed_loop_contracts import LoopConfig
from robot_control.closed_loop_live import check_qualification

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    check_qualification(args.evidence, LoopConfig.load(args.profile))
