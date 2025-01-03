# Speech Emotion Recognition System

This document outlines a comprehensive Speech Emotion Recognition (SER) system implemented using advanced machine learning techniques, specifically Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), Support Vector Machines (SVM), and Multi-Layer Perceptron (MLP) models in Keras. The system has been enhanced through improved feature extraction methods, achieving an accuracy of approximately **80%**.

## Overview of the Project

The SER system is designed to classify emotions from audio samples, utilizing various neural network architectures to analyze vocal patterns. The original version of the project can be found in the **First-Version** branch of the GitHub repository.

## Environment Setup

- **Python Version:** 3.8
- **Libraries:** 
  - Keras & TensorFlow 2
  - Scikit-learn
  - Joblib
  - Librosa
  - SciPy
  - Pandas
  - Matplotlib
  - NumPy

## Project Structure

The following directory structure is established for the project:

```
├── models/ // Contains model definitions
│   ├── common.py // Base class for all models
│   ├── dnn // Neural network implementations
│   │   ├── dnn.py // Base class for DNN models
│   │   ├── cnn.py // CNN implementation
│   │   └── lstm.py // LSTM implementation
│   └── ml.py // SVM and MLP implementations
├── extract_feats/ // Feature extraction scripts
│   ├── librosa.py // Feature extraction using Librosa
│   └── opensmile.py // Feature extraction using OpenSMILE
├── utils/ // Utility functions and scripts
│   ├── files.py // Dataset setup and management
│   ├── opts.py // Argument parsing utilities
│   └── plot.py // Graph plotting functions
├── config/ // Hyperparameter configuration files (.yaml)
├── features/ // Directory for storing extracted features
├── checkpoints/ // Directory for model weights storage
├── train.py // Training script
├── predict.py // Emotion prediction script for audio input
└── preprocess.py // Data preprocessing script for feature extraction and storage
```

## Datasets Used

1. **RAVDESS**: Contains around 1500 audio samples from 24 actors portraying eight emotions.
2. **SAVEE**: Comprises approximately 500 audio samples from four male actors expressing seven emotions.
3. **EMO-DB**: Features around 500 audio samples from ten actors (five male, five female) displaying seven emotions.
4. **CASIA**: Includes about 1200 audio samples from four actors (two male, two female) with six different emotions.

## Installation Instructions

To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

Optionally, install OpenSMILE for additional feature extraction capabilities.

## Configuration and Usage

### Configuration

Adjust parameters in the YAML configuration files located in the `configs/` directory. Supported OpenSMILE standard feature sets include:

- IS09_emotion: 384 features 
- IS10_paraling: 1582 features 
- IS11_speaker_state: 4368 features 
- IS12_speaker_trait: 6125 features 
- IS13_ComParE: 6373 features 
- ComParE_2016: 6373 features 

### Preprocessing Data

To extract features from your dataset, run:

```bash
python preprocess.py --config configs/example.yaml
```

This command will process audio files and store extracted features in designated formats.

### Training the Model

Ensure your dataset is organized with emotion-specific folders. Then, execute:

```bash
python train.py --config configs/example.yaml
```

### Making Predictions

To predict emotions from a new audio file after training, modify the `predict.py` script with the path to your audio file, then run:

```bash
python predict.py --config configs/example.yaml
```

## Visualization Functions

Several utility functions are provided to visualize results:

### Radar Chart

To display predicted probabilities in a radar chart:

```python
import utils

# Example usage:
data_prob = np.array([0.1, 0.2, 0.3, 0.4]) # Replace with actual probabilities
class_labels = ['Angry', 'Happy', 'Sad', 'Neutral']
utils.radar(data_prob, class_labels)
```

## Conclusion

This SER system leverages state-of-the-art deep learning techniques to classify emotions from speech effectively. By utilizing a combination of LSTM, CNN, SVM, and MLP models along with robust feature extraction methods, it aims to enhance applications in areas such as customer service and mental health analysis.

For further details or contributions, please refer to the project repository or contact me directly at niwinkumar7@gmail.com .
