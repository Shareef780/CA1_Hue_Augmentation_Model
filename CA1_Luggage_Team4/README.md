# C389 CA1 - People Counting Assistant, T1 Luggage (Team 4, C389-001)

The system flags a photograph when at least one person is carrying luggage, and counts the
luggage-carrying pedestrians moving left and moving right.

## Contents

| Folder | File | What it is |
|---|---|---|
| training | CA1_local_exp_01_hue.ipynb | The laptop notebook that trained the final model: yolo26n, 100 epochs, image size 640, batch 16, seed 0, fliplr 0.0, with the one change hsv_h = 0.5 (hue augmentation). |
| training | CA1_exp_01_hue_colab.ipynb | The same hue training run for Google Colab. It reads the Roboflow key from Colab Secrets. |
| training | train_hue_only.py | The same training run as a short script. |
| training | CA1_local_training_tuning.ipynb | The learning-rate and batch-size tuning notebook (SGD). |
| service | luggage_app.py | The Gradio service: validate_image, count_detections, build_reply. |
| service | RUN_DEMO.bat, SETUP_FIRST_TIME.bat, SELF_TEST.bat | Start the service, install it, and run its self-test on Windows. |
| service | CA1_Luggage_Deployment_Gradio.ipynb, CA1_Luggage_Deployment_Colab.ipynb | Notebook versions of the service (local and Colab backup). |
| service/model | best.pt, metrics.json | Weights of the final model (hue only) and its scores. |

## Final model

yolo26n with hue augmentation 0.5. On the 96 test photographs: precision 0.948, recall 0.971,
mAP@0.5 0.992, mAP@0.5:0.95 0.803. The service counts boxes at a confidence threshold of 0.40.

## Not in this repository

- The Roboflow API key. The notebooks ask for it when they run.
- The photographs and labels. They are deleted at the end of the module, as the brief requires.
