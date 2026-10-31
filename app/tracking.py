import argparse
import csv
import os
from collections import Counter, defaultdict, deque

import cv2
from ultralytics import YOLO

# feeder and waterer are part of the pen, not pigs
EQUIPMENT = {"Feeder", "Waterer"}
POSTURES = ["Lateral Lying", "Sternal Lying", "Sitting", "Standing"]
ACTIVITIES = {"Feeding": "Feeding", "Drinking": "Drinking", "NF": "Other", "": "Other"}

TRACKER_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pig_tracker.yaml")

COLOURS = {
    "Lateral Lying": (230, 160, 60),
    "Sternal Lying": (200, 90, 200),
    "Sitting": (60, 180, 240),
    "Standing": (80, 200, 80),
}


def split_label(label):
    # "Sternal Lying Drinking" -> ("Sternal Lying", "Drinking")
    for posture in POSTURES:
        if label.startswith(posture):
            rest = label[len(posture):].strip()
            return posture, ACTIVITIES.get(rest, rest)
    return label, "Other"


def draw_box(frame, box, text, colour):
    x1, y1, x2, y2 = map(int, box)
    scale = max(0.5, frame.shape[1] / 1400)
    thick = 1 if scale < 0.8 else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick + 1)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 6, y1), colour, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def track_video(model, video_path, out_dir, conf=0.4, imgsz=800, step=1, window=15,
                min_seconds=1.0, tracker=TRACKER_CFG):
    """Track every pig through a video and work out how long each one spends in each behaviour.

    step:   only run the model on every nth frame (faster, less smooth)
    window: number of recent frames used for a majority vote on each pig's label,
            which stops the label flickering between similar classes
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seconds_per_frame = step / fps

    video_out = os.path.join(out_dir, "tracked.mp4")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (width, height))

    csv_out = os.path.join(out_dir, "detections.csv")
    csv_file = open(csv_out, "w", newline="")
    rows = csv.writer(csv_file)
    rows.writerow(["frame", "time_s", "pig_id", "label", "smoothed_label", "conf", "x1", "y1", "x2", "y2"])

    recent = defaultdict(lambda: deque(maxlen=window))
    time_in = defaultdict(Counter)   # pig id -> smoothed label -> frames
    first_seen = {}
    frame_no = -1
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % step:
            continue
        processed += 1

        # persist=False on the first frame resets the tracker, so IDs start again for every video
        result = model.track(frame, persist=processed > 1, conf=conf, imgsz=imgsz, tracker=tracker,
                             verbose=False)[0]
        boxes = result.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

        for box, cls, score, pig_id in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), ids):
            label = model.names[cls]

            if label in EQUIPMENT or pig_id is None:
                draw_box(frame, box, label, (120, 120, 120))
                continue

            recent[pig_id].append(label)
            smoothed = Counter(recent[pig_id]).most_common(1)[0][0]
            time_in[pig_id][smoothed] += 1
            first_seen.setdefault(pig_id, frame_no)

            posture, activity = split_label(smoothed)
            text = f"#{pig_id} {posture}" + ("" if activity == "Other" else f" + {activity}")
            draw_box(frame, box, text, COLOURS.get(posture, (200, 200, 200)))

            rows.writerow([frame_no, round(frame_no / fps, 2), pig_id, label, smoothed, round(score, 3),
                           *[round(v, 1) for v in box]])

        writer.write(frame)

    cap.release()
    writer.release()
    csv_file.close()

    summary = build_summary(time_in, first_seen, fps, seconds_per_frame, min_seconds)
    summary["video_seconds"] = round(processed * seconds_per_frame, 1)
    return video_out, csv_out, summary


def build_summary(time_in, first_seen, fps, seconds_per_frame, min_seconds):
    pigs = []
    herd = Counter()

    for pig_id, labels in time_in.items():
        total = sum(labels.values()) * seconds_per_frame
        # very short tracks are usually a detection blip or an ID switch
        if total < min_seconds:
            continue

        posture_time = Counter()
        activity_time = Counter()
        for label, frames in labels.items():
            posture, activity = split_label(label)
            posture_time[posture] += frames * seconds_per_frame
            activity_time[activity] += frames * seconds_per_frame
        herd.update(posture_time)

        pigs.append({
            "id": pig_id,
            "first_seen_s": round(first_seen[pig_id] / fps, 1),
            "seconds": round(total, 1),
            "posture_pct": {p: round(100 * posture_time[p] / total) for p in POSTURES},
            "feeding_s": round(activity_time["Feeding"], 1),
            "drinking_s": round(activity_time["Drinking"], 1),
            "main_behaviour": labels.most_common(1)[0][0],
        })

    pigs.sort(key=lambda p: p["id"])
    herd_total = sum(herd.values()) or 1
    return {
        "pigs": pigs,
        "pig_count": len(pigs),
        "herd_posture_pct": {p: round(100 * herd[p] / herd_total) for p in POSTURES},
    }


def print_summary(summary):
    print(f"\n{summary['pig_count']} pigs tracked over {summary['video_seconds']} s")
    print("herd posture: " + ", ".join(f"{p} {v}%" for p, v in summary["herd_posture_pct"].items()))
    print(f"\n{'pig':>4} {'on screen':>10} {'lat. lying':>11} {'stern. lying':>13} {'sitting':>8} "
          f"{'standing':>9} {'feeding':>8} {'drinking':>9}")
    for p in summary["pigs"]:
        pct = p["posture_pct"]
        print(f"{p['id']:>4} {p['seconds']:>9}s {pct['Lateral Lying']:>10}% {pct['Sternal Lying']:>12}% "
              f"{pct['Sitting']:>7}% {pct['Standing']:>8}% {p['feeding_s']:>7}s {p['drinking_s']:>8}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track pigs in a video and summarise their behaviour")
    parser.add_argument("video")
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--out", default="outputs/cli")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--step", type=int, default=1, help="run the model on every nth frame")
    args = parser.parse_args()

    model = YOLO(args.model)
    video, table, summary = track_video(model, args.video, args.out, conf=args.conf, step=args.step)
    print_summary(summary)
    print(f"\nvideo: {video}\ncsv:   {table}")
