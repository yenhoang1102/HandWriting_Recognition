# Handwritten Text Recognition using CRNN and CTC

## Overview

This project implements a **Handwritten Text Recognition (HTR)** system that converts images of handwritten names into digital text using deep learning.

The model is based on a **CRNN architecture (Convolutional Recurrent Neural Network)** combined with **CTC Loss (Connectionist Temporal Classification)**, which allows the network to learn text sequences without character-level alignment.

The system processes handwritten images, extracts visual features using CNN layers, models sequence dependencies using BiLSTM layers, and decodes predictions using CTC decoding.

---

## Dataset

This project uses the **Handwriting Recognition Dataset (Kaggle)**.

Dataset structure:

```
Handwriting Recognition
│
├── train_v2
│   └── train
├── validation_v2
│   └── validation
├── test_v2
│   └── test
│
├── written_name_train_v2.csv
├── written_name_validation_v2.csv
```

Each CSV file contains:

| Column   | Description                   |
| -------- | ----------------------------- |
| FILENAME | Image filename                |
| IDENTITY | Ground truth handwritten text |

Example:

```
TRAIN_00001.jpg , JOHN
TRAIN_00002.jpg , DAVID
```

---

## Model Architecture

The model architecture follows the **CRNN pipeline**:

```
Input Image (128x32 grayscale)
        │
        ▼
Convolutional Neural Network (CNN)
        │
        ▼
Feature Map
        │
        ▼
Reshape to Sequence
        │
        ▼
Bidirectional LSTM
        │
        ▼
Dense + Softmax
        │
        ▼
CTC Loss
```

Layers used:

* Conv2D
* MaxPooling
* Dropout
* Bidirectional LSTM
* Dense Softmax
* CTC Loss Layer

---

## Training Configuration

| Parameter  | Value    |
| ---------- | -------- |
| Image Size | 128 × 32 |
| Batch Size | 32       |
| Epochs     | 8        |
| Optimizer  | SGD      |
| Loss       | CTC Loss |

---

## Evaluation Metrics

The model is evaluated using two common OCR metrics:

### Word Error Rate (WER)

Measures the difference between predicted text and ground truth at the **word level**.

### Character Error Rate (CER)

Measures prediction accuracy at the **character level**.

```
WER = (Substitutions + Insertions + Deletions) / Words
CER = (Substitutions + Insertions + Deletions) / Characters
```

Both metrics are implemented using the **jiwer library**.

---

## Project Structure

```
handwriting-recognition
│
├── handwriting.py
├── prediction_model_ocr
├── predictions.txt
├── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/handwriting-recognition.git
cd handwriting-recognition
```

Install required libraries:

```
pip install tensorflow
pip install opencv-python
pip install numpy
pip install pandas
pip install matplotlib
pip install jiwer
```

---

## Running the Project

Train the model:

```
python handwriting.py
```

After training, the script will:

1. Generate predictions on the validation set
2. Save results in `predictions.txt`
3. Evaluate the model using **WER** and **CER**

---

## Example Output

```
Ground truth: SARAMITO 	 Predicted: SARAMITO
Ground truth: THEO 	 Predicted: THEO
Ground truth: MORGANE 	 Predicted: MORGANE

```

Evaluation metrics:

```
Word Error Rate (WER): 0.2863
Character Error Rate (CER): 0.0758
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* JiWER

---

## Future Improvements

Possible improvements include:

* Data augmentation for handwritten images
* Beam search decoding instead of greedy decoding
* Transformer-based OCR models
* Training on larger handwriting datasets

---

## Author

Yen Hoang
AI / Machine Learning Student

---

## License

This project is for educational and research purposes.
