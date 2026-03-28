# 🎧 Audio Classification (CEG3004)

## 📌 Project Overview

This project implements an audio classification pipeline for Environmental Sound Classification using an ESC-50-derived dataset. The goal is to classify audio clips into one of 50 environmental sound classes while ensuring robustness under clean, noisy, and band-limited conditions.

---

## 🎯 Objectives

* Extract meaningful DSP-based audio features
* Train a machine learning model to classify environmental sounds
* Improve robustness against distortions such as noise and bandwidth limitations
* Build a reproducible and well-structured pipeline

---

## 📂 Dataset

The dataset is derived from ESC-50 and contains:

* 2,000 audio clips across 50 classes
* 40 clips per class
* Each clip is 5 seconds long (mono)

The submission dataset includes:

* Clean audio
* Noisy audio
* Band-limited audio

This setup evaluates both classification accuracy and robustness.

---

## ⚙️ Pipeline Overview

1. Audio loading
2. Preprocessing
3. Feature extraction
4. Feature matrix construction
5. Model training and validation
6. Prediction on submission set
7. CSV generation

---

## 🔧 Preprocessing

The following preprocessing steps were applied:

* Peak normalization to standardize amplitude
* Silence trimming using `librosa.effects.trim()`
* Fixed-length padding/truncation to 5 seconds

These steps ensure consistent input representation across all samples.

---

## 📊 Feature Extraction

A combination of DSP-based features was used:

### MFCC

* 20 Mel-Frequency Cepstral Coefficients
* Mean pooling across time

### Log-Mel Spectrogram

* Converted to decibel scale
* Mean pooling across time

### Spectral Features

* Spectral centroid
* Spectral bandwidth
* Spectral rolloff
* Zero-crossing rate

These features provide complementary information about timbre, frequency distribution, and signal characteristics.

---

## 🤖 Model

We used a Support Vector Machine (SVM) with an RBF kernel.

* Kernel: RBF
* C = 10
* gamma = 0.002
* class_weight = balanced

A StandardScaler was applied before training to normalize feature distributions.

### Why SVM?

Compared to Logistic Regression, the SVM with RBF kernel can model nonlinear relationships between audio features and classes, leading to improved performance.

---

## 🧪 Experiments

We evaluated multiple configurations of the SVM model:

| C   | Gamma | Macro-F1   |
| --- | ----- | ---------- |
| 9   | 0.002 | 0.6368     |
| 10  | 0.002 | **0.6371** |
| 100 | 0.002 | 0.6207     |
| 10  | 0.003 | 0.6193     |

### Observations

* Best performance achieved at **C = 10, gamma = 0.002**
* Increasing `C` to 100 reduced performance → indicates **overfitting**
* Slight variation (C=9 vs 10) produced similar results → **model stability**
* Increasing `gamma` reduced performance → **sensitivity to kernel width**

### Conclusion

The final model uses:

* **C = 10**
* **gamma = 0.002**

This configuration provides the best balance between fitting and generalization.

---

## 📈 Evaluation

The model was evaluated using:

* Classification report
* Macro-F1 score

Macro-F1 is used because it equally weights all classes in this multi-class problem.

---

## 📊 Visualizations

Waveform and spectrogram visualizations were used to analyze audio signals.

Key observations:

* Engine and helicopter sounds show strong low-frequency patterns
* Rain and fire appear as broadband noise
* Impulsive sounds (e.g., clapping) show sharp spikes

These visualizations helped guide feature selection.

---

## 🔍 Error Analysis

### Common Challenges

* Classes with similar frequency characteristics are harder to distinguish
* Noise and distortion reduce discriminative features
* Some classes have overlapping temporal patterns

### Example Confusions

* Engine vs helicopter (low-frequency similarity)
* Rain vs crackling fire (noise-like spectra)

### Model Behavior

The model was sensitive to SVM hyperparameters:

* Increasing `C` beyond 10 led to overfitting
* Increasing `gamma` reduced generalization

This highlights the importance of balancing model complexity and generalization.

---

## 📦 How to Run

### 1. Install dependencies

```bash
pip install numpy scipy pandas scikit-learn librosa soundfile tqdm
```

### 2. Run the notebook

Execute all cells in order (Step 1 → Step 10).

### 3. Outputs

* `GROUP_ID_model.joblib`
* `GROUP_ID_predictions.csv`

---

## 📁 Repository Structure

```
.
├── README.md
├── notebook.ipynb
├── requirements.txt
└── outputs/
    ├── GROUP_ID_model.joblib
    └── GROUP_ID_predictions.csv
```

---

## ✅ Reproducibility Notes

* Do not modify dataset structure or clip IDs
* Use the same preprocessing and feature extraction for training and submission
* Run notebook cells sequentially
* Ensure correct GROUP_ID before submission

---

## 🚀 Future Improvements

* Add data augmentation (noise, gain, filtering)
* Use richer temporal features
* Apply cross-validation
* Explore CNNs on spectrograms
* Improve class-wise error analysis

---

## 👨‍💻 Author

CEG3004 Project Group: Pr_1
