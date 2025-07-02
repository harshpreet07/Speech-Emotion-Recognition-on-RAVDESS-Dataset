# Speech Emotion Recognition Using RAVDESS Dataset

## Overview

This project implements a speech emotion recognition system using deep learning to classify emotions from audio recordings. The system leverages a Convolutional Neural Network (CNN) architecture to identify **8 distinct emotions**:
**neutral, calm, happy, sad, angry, fearful, disgust, and surprised**.

The project explores various machine learning approaches including **Decision Trees**, **Random Forests**, **Multi-Layer Perceptrons (MLP)**, and ultimately, a sophisticated **CNN model** that achieves the best performance.

---

## Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, which contains audio recordings of actors expressing different emotions.

### Structure

- **Training data**: Actors 1–19 (`Actor_01` to `Actor_19`)
- **Test data**: Actors 20–24 (`Actor_20` to `Actor_24`)

### Features

- Audio files are in **.wav** **format**
- File naming convention:
  `03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav`

#### Emotion Codes (3rd part of filename):

- `01`: neutral
- `02`: calm
- `03`: happy
- `04`: sad
- `05`: angry
- `06`: fearful
- `07`: disgust
- `08`: surprised

---

## Model Architecture

The final model is a **Convolutional Neural Network (CNN)** with **depthwise separable convolution blocks**.

### Feature Extraction Layers

- **Input**: 3-channel mel-spectrogram representation of audio
- **Initial Convolution**: 7×7 kernels with BatchNorm and ReLU activation

### Depthwise Separable Convolution Blocks:

- **Block 1**: Depthwise + Pointwise convolutions (64→128 channels)
- **Block 2**: Depthwise + Pointwise convolutions (128→256 channels)
- **Block 3**: Depthwise + Pointwise convolutions (256→512 channels)
- **Global Average Pooling**: Reduces spatial dimensions to 1×1

### Classifier

- **Fully Connected Layers**: 512 → 256 → 8 units
- **Dropout**: 0.5 rate for regularization
- **Output**: 8 classes corresponding to the emotions

---

## Implementation Details

### Audio Processing Pipeline

- Load audio using `torchaudio`
- Convert stereo to mono if necessary
- Extract mel-spectrogram features:
  - `n_fft=1024`
  - `hop_length=512`
  - `n_mels=64`
- Convert to decibel scale
- Pad or crop spectrograms to fixed dimensions
- Replicate to 3 channels for CNN input

### Training Strategy

- **Data Split**: 66:34 train:validation stratified split
- **Batch Size**: 32
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with initial learning rate 0.001
- **Learning Rate Schedule**: ReduceLROnPlateau (factor 0.5, patience 5)
- **Training Duration**: 70 epochs
- **Model Selection**: Based on best validation loss

---

## Results

### Test Set (Actors 20–24):

- **Accuracy**: 72.33%
- **Per-Emotion Performance**: Detailed precision, recall, and F1 scores for each emotion
- **Confusion Matrix**: Visualization showing common misclassifications
- **Per-Actor Analysis**: Shows performance variation due to speaking styles

---

## Project Structure

- `Initial_Models.ipynb`: Decision Trees, Random Forests, and MLP models
- `cnn-final-ee708-project.ipynb`: Final CNN implementation and training
- `Evaluation_Test_Data.ipynb`: Test dataset evaluation and metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `test_results.csv`: Classification results on test data

---

## Comparison of Models

| Model         | Validation Accuracy |
| ------------- | ------------------- |
| Decision Tree | ~42%                |
| Random Forest | ~64%                |
| MLP           | ~50%                |
| CNN           | ~79.4% ✅           |

---

## Usage

### Requirements

- `Python 3.8+`
- `PyTorch 1.8+`
- `torchaudio`
- `librosa`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tqdm`

### Training the Model

Instructions to train the model can be found in `cnn-final-ee708-project.ipynb`.

### Evaluating the Model

Run `Evaluation_Test_Data.ipynb` to compute final metrics and confusion matrix.

---

## Conclusion

This project demonstrates the effectiveness of CNNs for speech emotion recognition.The final model's architecture with **depthwise separable convolutions** provides a good balance between model complexity and performance.

> Some emotions (like _"angry"_ and _"surprised"_) are recognized more accurately than others (like _"neutral"_ and _"calm"_)—aligning with human perception where stronger emotions are often easier to identify.

### Future Improvements

- Data augmentation techniques
- Exploring attention mechanisms
- Incorporating transformer-based architectures
- Fine-tuning hyperparameters
- Using pretrained audio feature extractors

---
## Contributors


| Name                | Roll no. | Email Id                |
| ------------------- | -------- | ----------------------- |
| Aritra Ambudh Dutta | 230191   | aritraad23@iitk.ac.in   |
| Archita Goyal       | 230187   | architag23@iitk.ac.in   |
| Harshpreet Kaur     | 230464   | harshpreet23@iitk.ac.in |
| Suyash Kapoor       | 231066   | suyashk23@iitk.ac.in    |
| Saksham Verma       | 230899   | sakshamv@iitk.ac.in     |

This Project was completed the Course Project of the course **EE708** offered in **Semester 2024-25/II** at **Indian Institute of Technology (IIT), Kanpur** under Prof. **[Rajesh M. Hegde](https://rajeshmhegde.com/)**.
