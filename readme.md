# Drowsiness Detection System

A computer vision system to detect driver drowsiness using deep learning techniques to classify eye states (open or closed) in real-time video.

---

## Project Overview

This project implements a drowsiness detection system that monitors a person's eyes using a webcam and triggers an alert when signs of drowsiness are detected. The system uses a **Convolutional Neural Network (CNN)** to classify eye states as either "Open" or "Closed" and tracks consecutive frames of closed eyes to determine drowsiness.

---

## Features

- Real-time eye state classification using CNN
- Drowsiness alert system with visual indicators
- Comprehensive model evaluation tools
- Detailed statistical analysis and visualization
- Model trained specifically on eye states (open/closed)
- FPS counter for performance monitoring

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Drowsiness_Detection
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset** with the following structure:
    ```
    dataset/
      ├── train/
      │   ├── Open/
      │   └── Closed/
      └── test/
            ├── Open/
            └── Closed/
    ```

---

## Usage

### Training the Model

To train the drowsiness detection model:

```bash
python train.py
```

This will:

- Load and preprocess images from the training dataset
- Build and train a CNN model
- Save the best model to `drowsiness_eye_cnn_model.h5`
- Generate training/validation accuracy and loss plots

---

### Testing the Model

To evaluate the model performance:

```bash
python test.py
```

This will:

- Test the model on images from the test dataset
- Generate detailed statistics and visualizations
- Create a confusion matrix and prediction reports
- Save results to CSV files for further analysis

---

### Running the Detection System

To start the real-time drowsiness detection:

```bash
python drowsiness_detection.py
```

- The webcam will activate and begin monitoring
- Status information will be displayed on screen
- A drowsiness alert will trigger after detecting closed eyes for several consecutive frames
- Press `q` to exit the application

---

## Model Architecture

The system uses a CNN with the following architecture:

- 4 convolutional layers with increasing filters (16, 32, 64, 128)
- Max pooling, batch normalization, and dropout for regularization
- Dense layers with final softmax activation for binary classification
- Trained with categorical crossentropy loss and Adam optimizer

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Confusion matrix
- Detailed statistics by image directory
- Confidence scores per class
- Visualization of correct/incorrect predictions

---

## Dataset

The system requires images of eyes in two states:

- **Open** - eyes are fully open
- **Closed** - eyes are closed

For best results, the dataset should include diverse samples covering different lighting conditions, eye shapes, and demographics.

---

## License

[Specify your license here]

---

## Acknowledgments

[List any acknowledgments or resources that helped with the project]"# drowsiness_detection" 
