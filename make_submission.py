"""Regenerate the submission file from the provided match data.

    python make_submission.py --team <team_name>

Writes evaluation/results/<team_name>.csv, validates it with the official checker and scores it
against the public 5-minute ground truth. Deterministic: the same data gives the same file.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession.run import build, summarize, write  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="submission", help="team name; becomes the CSV name")
    args = ap.parse_args()

    labels, _ = build()
    summarize(labels)
    path = write(labels, args.team)
    print("wrote", os.path.relpath(path, HERE))
    subprocess.check_call([sys.executable, os.path.join(HERE, "evaluation", "validate_submission.py"),
                           path])
