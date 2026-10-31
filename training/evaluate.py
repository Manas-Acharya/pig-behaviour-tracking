import argparse

from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Evaluate the pig behaviour model on a dataset split")
parser.add_argument("--data", required=True, help="path to the Roboflow data.yaml")
parser.add_argument("--model", default="../app/weights/best.pt")
parser.add_argument("--split", default="test", choices=["val", "test"])
args = parser.parse_args()

model = YOLO(args.model)
metrics = model.val(data=args.data, split=args.split, imgsz=800, batch=8, plots=True)

box = metrics.box
print(f"\n{args.split} set: precision {box.mp:.3f}, recall {box.mr:.3f}, "
      f"mAP@0.5 {box.map50:.3f}, mAP@0.5:0.95 {box.map:.3f}\n")

print(f"{'class':<24} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}")
for i, cls in enumerate(box.ap_class_index):
    p, r, ap50, ap = box.class_result(i)
    print(f"{model.names[int(cls)]:<24} {p:>6.3f} {r:>6.3f} {ap50:>7.3f} {ap:>9.3f}")
