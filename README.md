# 🎵 Music Genre Classification Using GPU-Accelerated Deep Learning

This repository contains a PyTorch-based project for classifying audio files into different musical genres using a Convolutional Neural Network (CNN). The project leverages GPU acceleration to significantly speed up the training process of the deep learning model.

## Project Overview


## Why GPU?

Training deep learning models, especially those with many layers and parameters like CNNs, involves a massive amount of matrix multiplications and other parallelizable computations. CPUs (Central Processing Units) are designed for general-purpose tasks and excel at sequential processing. GPUs, on the other hand, are designed with thousands of smaller cores optimized for parallel processing.

In this project, the GPU is used for:

* **Accelerated Audio Preprocessing:** Operations like resampling audio and generating Mel spectrograms (which involve Fast Fourier Transforms and matrix operations) are performed directly on the GPU using `torchaudio` when the data is moved to the `cuda` device. This significantly speeds up the data loading and transformation pipeline.
* **Faster Model Training:** The core of deep learning training involves forward and backward passes through the neural network. All the convolutional operations, matrix multiplications in the fully connected layers, and gradient calculations during backpropagation are executed on the GPU. This parallel processing capability of the GPU allows for training the model much faster than on a CPU, enabling experimentation with more complex models or larger datasets.

## Code Architecture

The project follows a standard deep learning pipeline for audio classification:

1.  **Data Preprocessing **
    * Loads `.au` audio files.
    * Normalizes, resamples, pads/ trims audio.
    * Transforms audio into Mel spectrograms.
    * **GPU Usage:** Waveforms and Mel Spectrogram transform moved to GPU for accelerated feature extraction.
    Here's a visual representation of an audio file transformed into a Mel spectrogram, which is the input to our CNN model.
    ![Example Mel Spectrogram](https://github.com/ayush-kitnawat-2023160/Music_Gener_Classification/blob/f31dc2315c2e36cedb1ddf371536747cf3c6cad7/spectrograms/jazz/jazz.00083.png)

2.  **Model Definition **
    * A custom 2D Convolutional Neural Network (CNN).
    * Comprises three convolutional blocks (Conv2d, BatchNorm, ReLU, MaxPool) for feature learning from spectrograms.
    * Followed by fully connected layers with BatchNorm and Dropout for classification.
    * **GPU Usage:** The entire model is instantiated and moved to the GPU for all forward and backward passes.

3.  **Training & Evaluation Loop:**
    * Divides data into train/validation/test sets using `DataLoader`s.
    * `train_model`: Performs optimization (forward pass, loss calculation, backpropagation, weight update).
    * `evaluate_model`: Computes loss and accuracy on validation/test sets.
    * **GPU Usage:** All tensor operations within `train_model` and `evaluate_model` (including model inference, loss calculation, and gradient computations) are executed on the GPU.

4.  **Artifact Generation:**
    * Saves Mel spectrogram images for visual inspection.
    * Saves the best-performing model checkpoint (`best_model.pth`).
    * Plots and saves training/validation loss and accuracy curves.

## Getting Started

#### Prerequisites

* Python 3.8+
* NVIDIA GPU (recommended for acceleration, otherwise it will run on CPU)
* `pip` package manager

To run this project, follow these steps:

#### 1. Prerequisites

* **Python 3.8+**
* **Optional (for faster training):** NVIDIA GPU with CUDA support.

#### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ayush-kitnawat-2023160/Music_Gener_Classification.git
    cd audio-genre-classification
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: .\venv\Scripts\activate
    ```

3.  **Install PyTorch and Torchaudio:**
    * **With NVIDIA GPU (recommended):** Find the appropriate command for your CUDA version [here](https://pytorch.org/get-started/locally/). Example for CUDA 11.8:
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
        ```
    * **CPU-only:**
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
        ```

4.  **Install other dependencies:**
    ```bash
    pip install matplotlib scikit-learn tqdm
    ```

#### 3. Dataset Setup

1.  **Download the GTZAN Genre Collection dataset** (or a similar `.au` audio genre dataset).
2.  **Organize the dataset:** Create a `genres` folder in the project root. Inside `genres`, create subfolders for each genre (e.g., `blues/`, `classical/`), and place the corresponding `.au` audio files within.

    ```
    audio-genre-classification/
    ├── genres/
    │   ├── blues/
    │   │   ├── blues.00000.au
    │   │   └── ...
    │   └── classical/
    │       ├── classical.00000.au
    │       └── ...
    ├── main.py
    └── ...
    ```

#### 4. Run the Code

1.  **Ensure your virtual environment is active.**
2.  **Execute the main script:**
    ```bash
    python main.py
    ```
    Or
    ```bash
    make clean run
    ```
    The script will train the model, generate spectrograms in the `Spectrograms/` directory, save plots in `Plots/`, and log progress to `training_log.txt`.

### Model Performance

After training for 20 epochs, our Convolutional Neural Network demonstrated strong learning capabilities. The model's progress was consistently monitored, and the best version, based on validation accuracy, was saved.

**Key Training Highlights:**

* **Initial Learning (Epoch 1):** The model quickly moved from a low starting point (Train Acc: **35.25%**) to a respectable validation accuracy of **54.00%**.
* **Rapid Improvement (Epochs 2-7):** Significant gains were observed, with training accuracy soaring (e.g., Epoch 4 Train Acc: **78.12%**) and validation accuracy steadily climbing, reaching **71.00%** by Epoch 7.
* **Peak Performance (Epoch 18):** The model achieved its highest validation accuracy of **78.00%**, indicating effective generalization to unseen data at this point. While training accuracy continued to climb (reaching **99.38%**), the validation performance started to slightly fluctuate.

**Final Test Set Evaluation:**

The best-performing model (saved from Epoch 18) was evaluated on a completely unseen test dataset to assess its true generalization capability.

* **Test Loss:** **0.8818**
* **Overall Test Accuracy:** **76.00%**

This indicates that the model can correctly classify approximately 76% of new, unseen audio files into their respective genres.

**Detailed Classification Report (Test Set):**

This report includes precision, recall, and F1-score for each genre, highlighting where the model performs best and where there might be challenges.

| Genre       | Precision | Recall | F1-Score | Support |
| :---------- | :-------- | :----- | :------- | :------ |
| blues       | 0.73      | 0.85   | 0.79     | 13      |
| classical   | 0.80      | 1.00   | 0.89     | 8       |
| country     | 0.78      | 0.70   | 0.74     | 10      |
| disco       | 0.56      | 0.83   | 0.67     | 6       |
| hiphop      | 0.90      | 0.82   | 0.86     | 11      |
| jazz        | 0.91      | 0.62   | 0.74     | 16      |
| metal       | 0.91      | 0.91   | 0.91     | 11      |
| pop         | 0.89      | 0.73   | 0.80     | 11      |
| reggae      | 1.00      | 0.60   | 0.75     | 10      |
| rock        | 0.20      | 0.50   | 0.29     | 4       |
| **Accuracy**|           |        | **0.76** | **100** |
| **Macro Avg**| 0.77     | 0.76   | 0.74     | 100     |
| **Weighted Avg**| 0.82    | 0.76   | 0.77     | 100     |

#### Expected Output

* **Console Output/`log.txt`:** Training progress, loss and accuracy for each epoch, test set performance, and a detailed classification report.
* **`Spectrogram/` directory:** Contains subfolders for each genre, with `.png` images of Mel spectrograms for each audio file.
* **`best_model.pth`:** The saved weights of the model that achieved the highest validation accuracy during training.
* **`Plot/` directory:**
    * `loss_curves.png`: Plot of training and validation loss over epochs.
    * `accuracy_curves.png`: Plot of training and validation accuracy over epochs.

### Future Enhancements

* **More Advanced Models:** Experiment with recurrent neural networks (RNNs/LSTMs) or attention mechanisms combined with CNNs for potentially better performance on sequential audio data.
* **Data Augmentation:** Implement audio data augmentation techniques (e.g., time stretching, pitch shifting, adding noise) to increase the dataset size and improve model generalization.
* **Deployment:** Create a simple web interface or API for real-time genre prediction.

