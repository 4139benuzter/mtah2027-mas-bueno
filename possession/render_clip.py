"""Render a stretch of the tactical video with the submission's labels drawn on top.

    python -m possession.render_clip --csv "evaluation/results/Más Bueno.csv" \
        --start 11185 --end 18685 --out ../clips/first5min.mp4

Draws, for every frame: tracked players coloured by side (via the per-frame homography), the
tracked ball, the predicted label, the public ground truth where it exists, and a two-row
timeline (prediction over truth) with a cursor. Output is 1280x720 at the video's frame rate.
The clip contains match footage: it is for the team's own eyes, never for publishing.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402
from possession.atd import load_tracking, HOME  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402

COLOURS = {"H": (60, 60, 230), "A": (230, 140, 40), "D": (200, 200, 200), "X": (90, 90, 90)}
NAMES = {"H": "BARCA IN POSSESSION", "A": "ATHLETIC IN POSSESSION", "D": "DEAD BALL", "X": "no label"}
HOME_DOT, AWAY_DOT, BALL_DOT = (40, 40, 220), (230, 150, 40), (255, 255, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--start", type=int, default=11185)
    ap.add_argument("--end", type=int, default=18685)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    ap.add_argument("--ball-lead", type=int, default=3,
                    help="draw the tracked ball this many frames ahead: Metrica's ball track trails the "
                         "video slightly, and the model (and the ground truth) run on the tracker's clock")
    args = ap.parse_args()

    g = GAMES[GAME]
    fps = g["fps"]
    pred = pd.read_csv(args.csv)["label"].astype(str).str.strip().str.upper().values
    gt = pd.read_csv(args.gt)["label"].astype(str).str.strip().str.upper().values
    trk = load_tracking()
    length, width = g["pitch"]
    H, _, valid = load_homography()
    video_path = require(["video"])[0]

    cap = cv2.VideoCapture(video_path)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video: {} frames {}x{} (expected {})".format(n_video, src_w, src_h, g["n_video_frames"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    out_w, out_h = 1280, 720
    sx, sy = out_w / src_w, out_h / src_h
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    hm = trk["team"] == HOME
    agree = total = 0
    span = args.end - args.start
    bar_x0, bar_x1, bar_y = 40, out_w - 40, out_h - 60
    # The timeline strip never changes: draw it once and paste it on every frame.
    strip = np.full((out_h - (bar_y - 30), out_w, 3), 20, dtype=np.uint8)
    for k in range(bar_x0, bar_x1):
        ff = args.start + int((k - bar_x0) / (bar_x1 - bar_x0) * span)
        cv2.line(strip, (k, 8), (k, 24), COLOURS[pred[ff]], 1)
        cv2.line(strip, (k, 32), (k, 48), COLOURS[gt[ff]], 1)
    cv2.putText(strip, "model", (bar_x1 + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(strip, "truth", (bar_x1 + 4, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(strip, "red = Barca   orange = Athletic   grey = dead ball   (keepers are often not in the tracking)",
                (bar_x0, strip.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    for f in range(args.start, args.end):
        ok, frame = cap.read()
        if not ok:
            print("stopped early at frame", f)
            break
        frame = cv2.resize(frame, (out_w, out_h))

        # players and ball through the homography of this frame
        if valid[f]:
            P = trk["players"][f]
            for i in np.flatnonzero(~np.isnan(P[:, 0])):
                px, py = project(H[f], P[i, 0] / length, P[i, 1] / width)
                if np.isfinite(px) and np.isfinite(py):
                    cv2.circle(frame, (int(px * sx), int(py * sy)), 9, HOME_DOT if hm[i] else AWAY_DOT, 2)
            b = trk["ball"][min(n_video - 1, f + args.ball_lead)]
            if not np.isnan(b[0]):
                bx, by = project(H[f], b[0] / length, b[1] / width)
                if np.isfinite(bx) and np.isfinite(by):
                    cv2.circle(frame, (int(bx * sx), int(by * sy)), 7, BALL_DOT, -1)
                    cv2.circle(frame, (int(bx * sx), int(by * sy)), 7, (0, 0, 0), 1)

        # banner
        p, t = pred[f], gt[f]
        cv2.rectangle(frame, (0, 0), (out_w, 92), (20, 20, 20), -1)
        cv2.putText(frame, "MAS BUENO: " + NAMES[p], (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOURS[p], 2, cv2.LINE_AA)
        if t in ("H", "A", "D"):
            total += 1
            agree += int(p == t)
            mark = "match" if p == t else "MISS"
            cv2.putText(frame, "truth: " + NAMES[t] + "   " + mark, (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        COLOURS[t] if p == t else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "frame agreement so far {:.0%}".format(agree / total), (out_w - 420, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "truth: not published for this frame", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (160, 160, 160), 2, cv2.LINE_AA)
        cv2.putText(frame, "frame {}  {:02d}:{:02d}".format(f, (f // fps) // 60, (f // fps) % 60), (out_w - 420, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

        # timeline: prediction row over truth row, cursor at the current frame
        frame[bar_y - 30:, :] = strip
        kx = bar_x0 + int((f - args.start) / span * (bar_x1 - bar_x0))
        cv2.line(frame, (kx, bar_y - 26), (kx, bar_y + 22), (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    cap.release()
    print("wrote {} ({} frames); frame agreement with truth on graded frames: {:.1%}".format(
        args.out, args.end - args.start, agree / total if total else float("nan")))


if __name__ == "__main__":
    main()
