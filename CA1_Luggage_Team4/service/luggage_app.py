"""CA1 Luggage Direction Assistant, running on this computer.

No GPU and no internet are needed once the packages are installed.

    python luggage_app.py              open the page in the browser
    python luggage_app.py --selftest   error cases, sample photos and timings, for the report
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # nothing leaves this computer
os.environ.setdefault("YOLO_AUTOINSTALL", "False")           # never pip-install in the middle of a demo

from PIL import Image, ImageDraw, UnidentifiedImageError
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
WEIGHTS = os.path.join(MODEL_DIR, "best.pt")
METRICS = os.path.join(MODEL_DIR, "metrics.json")
SAMPLES = os.path.join(HERE, "sample_images")

if not os.path.exists(WEIGHTS):
    sys.exit("Cannot find %s. Keep the model folder next to luggage_app.py." % WEIGHTS)

# ---------------------------------------------------------------- the model, loaded once
MODEL = YOLO(WEIGHTS)
CLASS_NAMES = [MODEL.names[key] for key in sorted(MODEL.names)]

CONFIDENCE_THRESHOLD = 0.40
if os.path.exists(METRICS):
    with open(METRICS, encoding="utf-8") as handle:
        CONFIDENCE_THRESHOLD = float(json.load(handle).get("confidence_threshold", 0.40))

MAX_IMAGE_PIXELS = 40_000_000
SERVICE_VERSION = "1.0.0"

absent = [n for n in ("luggage_left", "luggage_right") if n not in CLASS_NAMES]
if absent:
    sys.exit("These weights do not know %s. They are not the luggage model." % absent)


# ---------------------------------------------------------------- the three functions
def validate_image(uploaded):
    """Check that what arrived can be turned into an image to run on.

    Returns (image, error), exactly one of which is None. The error is a dict with a
    status code and a message the caller can act on. Four cases are refused before any
    inference runs: nothing sent, an empty file, a file that is not a readable image, and
    an image so large it would exhaust memory.
    """
    if uploaded is None:
        return None, {"status": 400, "error": "No image was sent.",
                      "detail": "Attach a JPEG or PNG and send it again."}

    # Pillow's size guard would refuse huge images itself; this service applies its own limit.
    Image.MAX_IMAGE_PIXELS = None

    try:
        if isinstance(uploaded, Image.Image):
            image = uploaded
        elif isinstance(uploaded, (bytes, bytearray)):
            if not uploaded:
                return None, {"status": 400, "error": "The upload was empty.",
                              "detail": "The file carried no data."}
            image = Image.open(io.BytesIO(uploaded))
        else:
            if os.path.getsize(uploaded) == 0:
                return None, {"status": 400, "error": "The upload was empty.",
                              "detail": "The file '%s' is zero bytes."
                                        % os.path.basename(str(uploaded))}
            image = Image.open(uploaded)

        # The size is known from the header, so refuse before the pixels are decoded.
        if image.width * image.height > MAX_IMAGE_PIXELS:
            return None, {"status": 413, "error": "The image is too large to process.",
                          "detail": "%d by %d pixels is beyond the limit of %d pixels."
                                    % (image.width, image.height, MAX_IMAGE_PIXELS)}
        image.load()
    except Exception:
        # Ultralytics patches Image.open to try HEIF on failure, which raises ImportError
        # rather than UnidentifiedImageError, so every failure to read is a 415 here.
        return None, {"status": 415, "error": "That file is not a readable image.",
                      "detail": "Send a JPEG or PNG."}

    return image.convert("RGB"), None


def count_detections(image, threshold):
    """Run the model on one image and total the detections by class.

    Returns (counts, kept). Every class appears in counts, zeros included, and an image
    with nothing in it is a valid result of zero rather than an error.
    """
    result = MODEL.predict(image, conf=0.01, verbose=False, device="cpu")[0]

    kept = []
    boxes = result.boxes
    if boxes is not None:
        for index in range(len(boxes)):
            confidence = round(float(boxes.conf[index].item()), 3)
            if confidence < threshold:
                continue
            class_id = int(boxes.cls[index].item())
            kept.append({
                "class_name": CLASS_NAMES[class_id],
                "confidence": confidence,
                "box": [round(float(v), 1) for v in boxes.xyxy[index].tolist()],
            })

    counts = {name: 0 for name in CLASS_NAMES}
    for detection in kept:
        counts[detection["class_name"]] += 1

    return counts, kept


def build_reply(uploaded, threshold=CONFIDENCE_THRESHOLD, quiet=False):
    """The whole service in one function. Validate, count, answer.

    Returns (reply, image). A reply carrying "error" failed and one carrying "counts" did
    not, so a caller never has to read prose to tell them apart.
    """
    image, error = validate_image(uploaded)
    if error is not None:
        return error, None

    started = time.time()
    counts, detections = count_detections(image, threshold)
    elapsed = time.time() - started

    left = counts.get("luggage_left", 0)
    right = counts.get("luggage_right", 0)
    reply = {
        "status": 200,
        "counts": counts,
        # What this assistant exists for: people carrying luggage, by direction. Each
        # luggage box covers one carrier, so the total is the count of carrying
        # pedestrians and "flagged" says whether there is at least one.
        "luggage": {"left": left, "right": right, "total": left + right,
                    "flagged": (left + right) > 0},
        "confidence_threshold": threshold,
        "model_version": SERVICE_VERSION,
        "image_size": [image.width, image.height],
        "inference_seconds": round(elapsed, 3),
        "detections": detections,
    }
    if not quiet:
        print("counted %s in %.3fs on a %dx%d image" % (counts, elapsed, image.width, image.height))
    return reply, image


# ---------------------------------------------------------------- the page
BOX_COLOURS = {
    "luggage_left": (230, 120, 0),      # orange
    "luggage_right": (0, 150, 136),     # teal
    "person_left": (214, 69, 65),       # red
    "person_right": (46, 112, 196),     # blue
}


def annotate(image, detections):
    """Draw the kept boxes, luggage thicker than people, so the reply can be checked by eye."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for detection in detections:
        name = detection["class_name"]
        left, top, right, bottom = detection["box"]
        colour = BOX_COLOURS.get(name, (120, 120, 120))
        width = 5 if name.startswith("luggage") else 2
        draw.rectangle([left, top, right, bottom], outline=colour, width=width)
        draw.text((left + 3, max(0, top - 12)),
                  "%s %.2f" % (name, detection["confidence"]), fill=colour)
    return marked


def predict(uploaded, threshold):
    """What the page calls. Returns the annotated image and the reply."""
    reply, image = build_reply(uploaded, threshold)
    if image is None:
        return None, reply
    return annotate(image, reply["detections"]), reply


def launch():
    import gradio as gr

    examples = []
    if os.path.isdir(SAMPLES):
        examples = [[os.path.join(SAMPLES, n), CONFIDENCE_THRESHOLD]
                    for n in sorted(os.listdir(SAMPLES)) if n.lower().endswith((".jpg", ".jpeg", ".png"))]

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Walkway photograph"),
            gr.Slider(0.05, 0.95, value=CONFIDENCE_THRESHOLD, step=0.05, label="Confidence threshold"),
        ],
        outputs=[gr.Image(label="Detections"), gr.JSON(label="Reply")],
        title="Luggage Direction Assistant",
        description="Upload one photograph. The reply says how many people carrying luggage "
                    "are moving left and how many are moving right. Orange boxes are luggage "
                    "moving left, teal boxes are luggage moving right.",
        examples=examples or None,
        cache_examples=False,
    )
    print("")
    print("Open the page that appears in your browser. If it does not, copy the address below.")
    print("Press Ctrl+C in this window to stop the service.")
    demo.launch(server_name="127.0.0.1", inbrowser=True)


# ---------------------------------------------------------------- the self test
def selftest():
    print("model      :", WEIGHTS)
    print("classes    :", CLASS_NAMES)
    print("threshold  :", CONFIDENCE_THRESHOLD)

    # 1. the error conditions, called on build_reply directly
    scratch = tempfile.mkdtemp()
    text_file = os.path.join(scratch, "not_an_image.txt")
    with open(text_file, "w", encoding="utf-8") as handle:
        handle.write("this is not a photograph")
    empty_file = os.path.join(scratch, "empty.jpg")
    open(empty_file, "wb").close()
    huge_file = os.path.join(scratch, "huge.jpg")
    Image.new("RGB", (9000, 7000), (200, 200, 200)).save(huge_file, quality=30)
    nobody_file = os.path.join(scratch, "nobody.jpg")
    Image.new("RGB", (640, 480), (214, 210, 202)).save(nobody_file)

    cases = [("nothing sent", None), ("a text file", text_file), ("an empty file", empty_file),
             ("a very large image", huge_file), ("an image with no luggage in it", nobody_file)]
    print("\n--- 1. error conditions ---")
    print("%-32s %-7s %s" % ("case", "status", "what came back"))
    for label, payload in cases:
        reply, _ = build_reply(payload, quiet=True)
        print("%-32s %-7s %s" % (label, reply["status"], reply.get("error") or ("counts " + str(reply.get("counts")))))

    # 2. the sample photographs against the hand count
    truth_path = os.path.join(SAMPLES, "truth.json")
    if os.path.exists(truth_path):
        with open(truth_path, encoding="utf-8") as handle:
            truth = json.load(handle)
        print("\n--- 2. sample photographs: hand count (from the labels) against the service ---")
        print("%-26s %-16s %-16s %s" % ("photo", "annotated L / R", "service L / R", ""))
        same = 0
        for name in sorted(truth):
            path = os.path.join(SAMPLES, name)
            if not os.path.exists(path):
                continue
            reply, _ = build_reply(path, quiet=True)
            want = (truth[name]["luggage_left"], truth[name]["luggage_right"])
            got = (reply["luggage"]["left"], reply["luggage"]["right"])
            same += want == got
            print("%-26s %-16s %-16s %s" % (name[:25], "%d / %d" % want, "%d / %d" % got,
                                           "" if want == got else "<-- differs"))
        print("%d of %d agree" % (same, len(truth)))

    # 3. warm and cold response time
    sample = Image.new("RGB", (640, 480), (214, 210, 202))
    count_detections(sample, CONFIDENCE_THRESHOLD)          # the first call pays one-off set-up
    warm = []
    for _ in range(5):
        started = time.time()
        count_detections(sample, CONFIDENCE_THRESHOLD)
        warm.append(time.time() - started)
    warm_average = sum(warm) / len(warm)

    probe_path = os.path.join(scratch, "cold_start_probe.py")
    with open(probe_path, "w", encoding="utf-8") as handle:
        handle.write("import sys, time\nlaunch = float(sys.argv[1])\nstart = time.time()\n"
                     "from ultralytics import YOLO\nfrom PIL import Image\nimported = time.time()\n"
                     "model = YOLO(%r)\nloaded = time.time()\n"
                     "model.predict(Image.new('RGB', (640, 480)), conf=0.01, verbose=False, device='cpu')\n"
                     "answered = time.time()\n"
                     "print('  importing the libraries : %%.3f s' %% (imported - start))\n"
                     "print('  reading the weights     : %%.3f s' %% (loaded - imported))\n"
                     "print('  answering the first time: %%.3f s' %% (answered - loaded))\n"
                     "print('ANSWERED %%.3f' %% (answered - launch))\n" % WEIGHTS)
    started = time.time()
    probe = subprocess.run([sys.executable, probe_path, repr(started)], capture_output=True, text=True)
    cold_total, lines = time.time() - started, []
    for line in probe.stdout.splitlines():
        if line.startswith("ANSWERED "):
            cold_total = float(line.split()[1])
        elif line.strip():
            lines.append(line)
    print("\n--- 3. response time on this computer (CPU) ---")
    for line in lines:
        print(line)
    print("a caller reaching a cold service waits : %.3f s" % cold_total)
    print("a caller reaching a warm one waits     : %.3f s" % warm_average)
    print("cold is %.0f times the warm figure" % (cold_total / max(warm_average, 1e-9)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="run the checks instead of the page")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        launch()
