# Day 12: Convert a Classifier to TensorFlow Lite

This project trains a small neural network to classify light-level readings as **Dark**, **Normal**, or **Bright**. It converts the trained model to TensorFlow Lite (`.tflite`) and checks predictions from the converted model against the original model in Python.

## Project files

- `python.py` — trains the classifier, converts it to TensorFlow Lite, and compares predictions.
- `light_classifier.tflite` — converted TensorFlow Lite model.
- `sketch.ino` and `diagram.json` — Wokwi ESP32 simulation files, if included in this repository.

## Wokwi Simulation

[Open the Wokwi simulation](https://wokwi.com/projects/477019075940449281)

## Run the Python project

1. Open `python.py` in Google Colab or run it in a Python environment with TensorFlow installed.
2. Run the script to train the classifier and create the `.tflite` model.
3. Review the test predictions and confirm the original and TFLite model outputs match.

## Light-level classes

- **Dark** — low light reading
- **Normal** — medium light reading
- **Bright** — high light reading

The exact reading ranges depend on the dataset and thresholds used in `python.py`.
