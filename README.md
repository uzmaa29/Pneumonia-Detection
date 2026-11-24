# Pneumonia Detection from Chest X-Rays

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA**. It has a Keras CNN trained from scratch, a Flask web app that serves it, and an earlier fastai ResNet50 notebook for reference.

> **For educational use only.** This is not a medical tool and must not be used to diagnose pneumonia or guide treatment.

**Video demonstration:** [v]

## Repository contents

| File | Purpose |
| --- | --- |
| `notebooks/pneumonia_cnn.ipynb` | Keras CNN: data loading, patient-grouped split, training, test evaluation (Colab) |
| `models/pneumonia_cnn.keras` | Trained Keras model used by the Flask app |
| `app.py` | Flask app: upload page and `/predict` route |
| `templates/index.html` | Upload page |
| `requirements.txt` | Python packages for the Flask app |
| `notebooks/pneumonia_prediction.ipynb` | fastai ResNet50 notebook (earlier experiment, for reference) |

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney on Kaggle. See the dataset page for its license and terms.

## Keras CNN method

- **Input:** 224 x 224 RGB images, scaled to 0 to 1.
- **Model:** three Conv2D blocks (32, 64, 128 filters), each with BatchNormalization and MaxPooling, then Flatten, Dense(128), Dropout(0.5), and a sigmoid output.
- **Augmentation:** small rotation, zoom, and horizontal flip, applied during training only.
- **Class imbalance:** class weights computed from the training set.
- **Split:** validation is taken from `train/` with a patient-grouped split, so no patient appears in both training and validation. The `test/` folder is used once, for the final evaluation.
- **Training:** Adam (learning rate 1e-3), binary cross-entropy, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.

Split sizes: 4,163 training, 1,053 validation, and 624 test images.

## Results

Trained on Google Colab with a T4 GPU. The saved weights are from epoch 6, the best validation loss.

**Test set (624 images, not used for training or tuning):**

| Metric | Value |
| --- | --- |
| Accuracy | 86.5% |
| AUC | 0.925 |
| NORMAL recall | 88% (207 of 234) |
| PNEUMONIA recall | 85% (333 of 390) |

Confusion matrix (rows are true labels):

| | Predicted NORMAL | Predicted PNEUMONIA |
| --- | --- | --- |
| **True NORMAL** | 207 | 27 |
| **True PNEUMONIA** | 57 | 333 |

An earlier fastai run reported 97.57%, but that split put images from the same patients into both training and validation, so it overstated performance. The numbers above are the reliable ones.

## Limitations

- Results come from one dataset. They haven't been checked on data from other hospitals, scanners, or patient groups.
- Validation accuracy swung a lot between epochs, so results can change between runs.
- The model misses some pneumonia cases (57 in the test set). Missed cases matter most in a clinical setting.
- The Flask app is for local demonstration only. It has no authentication and no logging.

## Run the Flask app

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, upload a chest X-ray, and click **Predict**.

## Train the model

1. Open `notebooks/pneumonia_cnn.ipynb` in Google Colab.
2. Select a GPU runtime (Runtime → Change runtime type → T4 GPU).
3. Put your Kaggle API key in `kaggle.json` at `/content/`. Don't commit this file.
4. Run the cells in order. The notebook downloads the dataset, trains the model, evaluates on the test set, and saves `pneumonia_cnn.keras`. Move that file to `models/` before running the app.

## Tech stack

Python · TensorFlow / Keras · Flask · NumPy · Pillow · pandas · scikit-learn · fastai · Google Colab · Kaggle API
