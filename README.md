# Pig Behaviour Tracking

**Detects pigs in farm camera footage, recognises what each one is doing, and tracks every pig through a video to report how long it spends lying, sitting, standing, feeding and drinking.**

Built with YOLOv8, ByteTrack and FastAPI.

![Python](https://img.shields.io/badge/Python-3-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple) ![FastAPI](https://img.shields.io/badge/FastAPI-web%20app-009688)

---

## Why

Changes in how pigs lie, eat and drink are often the first sign of illness, heat stress or a feeding problem, but nobody can watch a pen all day. This project turns a camera feed into per-pig behaviour data that a farmer or vet can check at a glance.

## What It Does

- **Detects and classifies** each pig into one of 10 behaviours, combining a posture (lateral lying, sternal lying, sitting, standing) with an activity (feeding, drinking or neither). It also detects the feeder and the water trough.
- **Tracks each pig** through a video with ByteTrack, so every pig keeps the same ID from frame to frame.
- **Smooths the labels** with a majority vote over each pig's last 15 frames, so a pig's label does not flicker between similar classes.
- **Summarises behaviour per pig:** time on screen, share of time in each posture, and seconds spent feeding and drinking, plus a breakdown for the whole pen.
- **Web app:** upload an image or a video and get back the annotated result, the behaviour summary and a frame-by-frame CSV.
- **Command line:** `python tracking.py video.mp4` prints the same summary.

## Results

### Detection

YOLOv8s fine-tuned at 800x800, evaluated on the validation set (182 images, 2,188 labelled objects):

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference speed |
|---|---|---|---|---|
| 0.799 | 0.776 | 0.817 | 0.721 | 5.8 ms per image (T4 GPU) |

Common behaviours and the pen equipment are detected reliably: in validation, the feeder and water trough reached 0.99 mAP@0.5, and lateral lying and standing feeding about 0.90. Rare behaviours are much weaker. Sitting Drinking, with only 5 validation examples, reached 0.39, and Sternal Lying Feeding 0.64. Run `training/evaluate.py` to get the full per-class results on the held-out test set.

### Tracking

| Footage | Pigs in view | Track IDs created | Result |
|---|---|---|---|
| Fixed overhead camera, indoor pen | 6 | 6 | Every pig kept one ID for the whole clip |
| Handheld phone, outdoor pen | 8 to 13 | 45 | IDs fragment |

On footage that matches the training data, tracking is stable, and smoothing cut label changes by 59% (from 17 to 7 across the six pigs). On handheld footage the IDs break up. Switching to BoT-SORT, which corrects for camera motion, did not help (47 IDs). The cause is the detector: it was trained only on fixed overhead cameras, so on side-on, moving footage it misses pigs in some frames, and each pig that reappears gets a new ID.

---

## How the Tracking Works

1. **Detect:** YOLOv8s finds every pig, the feeder and the water trough in the frame.
2. **Link:** ByteTrack matches detections to existing tracks. Lost tracks are kept for 90 frames (the default is 30), because pigs often lie on top of each other and disappear briefly.
3. **Smooth:** each pig's label is the most common label among its last 15 frames.
4. **Split:** each label is split into posture and activity, for example "Sternal Lying Drinking" becomes "Sternal Lying" and "Drinking", and time is added to each.
5. **Filter:** tracks shorter than one second are dropped, since they are usually a false detection or an ID switch.
6. **Output:** an annotated video, a per-pig summary and a CSV with one row per pig per frame.

## Model Training

- **Data:** the [Pig Behavior dataset](https://universe.roboflow.com/swine-tktu8/pig-behavior-8xbgn/dataset/1) from Roboflow Universe (CC BY 4.0): overhead images of an indoor pen, split into 2,622 training, 182 validation and a held-out test set.
- **Classes:** Lateral Lying; Sitting Drinking, Feeding and NF; Standing Drinking, Feeding and NF; Sternal Lying Drinking, Feeding and NF; plus Feeder and Waterer. "NF" means not feeding.
- **Model:** YOLOv8s (11 million parameters) pretrained on COCO.
- **Training:** 800x800 images, batch size 8, on a Colab T4 GPU. The first 50-epoch run reached 0.828 mAP@0.5 and 0.721 mAP@0.5:0.95 on validation. Two further fine-tuning runs (20 and 30 epochs) did not improve on that, which suggests the model had converged. The app uses the 20-epoch checkpoint, which matches it on mAP@0.5:0.95.

The training notebook is in `training/train_yolov8.ipynb`.

## Limitations

- **Fixed overhead cameras only.** The model has only seen one camera angle in one type of pen, so other farms would need some of their own labelled footage for fine-tuning.
- **Rare behaviours.** Classes with few training examples, such as sitting while drinking, are detected much less reliably.
- **ID switches.** When two pigs overlap for a long time, or a pig leaves the frame, it can come back with a new ID, which splits its time across two tracks.
- **Speed on a CPU.** Processing runs at about 0.6 seconds per frame on a 2-core CPU, so the app processes every second frame by default (see `VIDEO_STEP` below).
- **Activity comes from the label alone.** Feeding and drinking are taken from the model's class rather than from checking that the pig is actually at the feeder or trough. Using the detected feeder and trough positions would be a natural next step.

## Future Work

- Decide feeding and drinking from each pig's position relative to the detected feeder and trough, which would also help the rare classes.
- Compare YOLOv8n, YOLOv8s and YOLO11 for accuracy against CPU speed.
- Read live RTSP camera streams and log behaviour over hours or days.
- Deploy a public demo.

---

## Running It

You need Python 3.9 or later and [FFmpeg](https://ffmpeg.org/download.html) on your PATH (used to make the output video play in the browser).

```bash
git clone https://github.com/Manas-Acharya/pig-behaviour-tracking.git
cd pig-behaviour-tracking/app
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000, upload an image or a video, and press **Analyze Behavior**.

**Settings** (environment variables):
- `MODEL_PATH`: path to the model weights (default `weights/best.pt`)
- `VIDEO_STEP`: run the model on every nth frame (default `2`; set to `1` if you have a GPU)

**Command line:**

```bash
cd app
python tracking.py path/to/video.mp4 --step 1 --out outputs/my_video
```

**Evaluate on the test set:** download the dataset from Roboflow in YOLOv8 format, then:

```bash
cd training
python evaluate.py --data path/to/pig_dataset/data.yaml --split test
```

## Repository Structure

```
pig-behaviour-tracking/
├── app/
│   ├── main.py               # FastAPI app: image detection and video tracking endpoints
│   ├── tracking.py           # Tracking, label smoothing and behaviour summary
│   ├── pig_tracker.yaml      # ByteTrack settings
│   ├── templates/index.html  # Web interface
│   ├── weights/best.pt       # Trained YOLOv8s model
│   └── requirements.txt
└── training/
    ├── train_yolov8.ipynb    # Model training (Google Colab)
    └── evaluate.py           # Per-class evaluation on the val or test split
```

## Background

This started as a group project during my MSc at the University of Essex, which produced the detection model and the first version of the web app. I later extended it on my own with multi-pig tracking, label smoothing, per-pig behaviour summaries, CSV export, image and video summaries in the web app, and a fix that stops simultaneous uploads overwriting each other.

## Credits and Licence

- Dataset: [Pig Behavior](https://universe.roboflow.com/swine-tktu8/pig-behavior-8xbgn) by the swine-tktu8 workspace on Roboflow Universe, licensed under CC BY 4.0.
- Detection and tracking: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Because Ultralytics is licensed under AGPL-3.0, this project uses the same licence.

---

**Author:** Manas Acharya, MSc Artificial Intelligence (Distinction), University of Essex
[LinkedIn](https://linkedin.com/in/manas-acharya-8m2a) · [GitHub](https://github.com/Manas-Acharya)
