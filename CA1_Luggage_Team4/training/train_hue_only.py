# Hue-only run for CA1 (Team 4): the baseline recipe with one change, hsv_h 0.5 instead of 0.015.
# Rebuilt from the saved args.yaml of the hue run (Ultralytics 8.4.120). Every setting not listed here
# is left at the Ultralytics 8.4.120 default, exactly as in the original run.
#
# Change DATA to where your dataset's data.yaml is. The original run used
# C:\C389_local\runs\CA1_experiments\dataset_by_scene\data.yaml (scene-level split, seed 0).
from ultralytics import YOLO

DATA = r"C:\C389_local\runs\CA1_experiments\dataset_by_scene\data.yaml"

model = YOLO("yolo26n.pt")
model.train(
    data=DATA,
    epochs=100,
    imgsz=640,
    batch=16,
    seed=0,
    workers=0,        # 0 on Windows
    fliplr=0.0,       # never flip: left and right are the classes
    bgr=0.0,
    hsv_h=0.5,        # THE ONE CHANGE (baseline is 0.015): wide hue shifts, so colour cannot decide the side
    hsv_s=0.7,        # defaults, shown for reference
    hsv_v=0.4,
    optimizer="auto", # Ultralytics picked AdamW in the original run
    name="hue",
)

# Scores on the held-out photos (best.pt is saved under runs/detect/hue/weights/)
best = YOLO("runs/detect/hue/weights/best.pt")
best.val(data=DATA, split="val")
best.val(data=DATA, split="test")
