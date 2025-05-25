# Machine Learning Course GUI

A comprehensive GUI application for machine learning experimentation and education, featuring classical ML algorithms, deep learning architectures, GANs, dimensionality reduction, reinforcement learning, and feature extraction.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
  - [Data Management](#data-management)
  - [Classical Machine Learning](#classical-machine-learning)
  - [Deep Learning](#deep-learning)
  - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Feature Extraction](#feature-extraction)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Visualization & Metrics](#visualization--metrics)
- [Logging & Troubleshooting](#logging--troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Classical Machine Learning**: Linear Regression, Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, K-Nearest Neighbors
- **Deep Learning**: Build custom MLP, CNN, RNN (LSTM/GRU) architectures layer-by-layer
- **Generative Adversarial Network (GAN)**: Generator & Discriminator with real-time progress and PCA visualization of generated samples
- **Dimensionality Reduction**: PCA, Truncated SVD, t-SNE, Linear Discriminant Analysis (LDA)
- **Feature Extraction**: Interactive component selection, explained variance plots
- **Reinforcement Learning**: Q-Learning, SARSA, DQN on standard Gym environments
- **Data Management**: Load built-in datasets (Iris, Boston Housing, Breast Cancer), or custom CSV with target selection; handling missing data and scaling
- **Visualization**: 3D scatter, histograms, training curves, prediction vs. actual plots
- **Real-time Logging**: Logs to file and GUI text pane via custom handler
- **Cross-Validation**: K-Fold accuracy & RMSE reports
- **GPU/CPU Support**: Leverages TensorFlow backend

---

## Prerequisites

- **Python**: 3.7 or higher
- **Dependencies** (see `requirements.txt`):
  - PyQt6
  - TensorFlow 2.4+
  - scikit-learn
  - matplotlib
  - numpy
  - pandas

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

```bash
python 20067502.py
```

> **Note**: The main script filename may vary (`20067502.py`). Ensure you run the correct file.

---

## Usage Guide

### Data Management

1. Select a dataset: **Iris**, **Boston Housing**, **Breast Cancer**, or **Custom CSV**.
2. Configure missing value handling: _No Action_ or _Mean Imputation_.
3. Choose scaling: _No Scaling_, _Standard_, _Min-Max_, _Robust_.
4. Set train/validation/test split or K-Fold value.
5. Click **Load Data**.

### Classical Machine Learning

- Navigate to **Classical ML** tab.
- Expand algorithm group, adjust hyperparameters.
- Click **Train**.
- View K-Fold results or final metrics.

### Deep Learning

- Go to **Deep Learning** tab:
  - **MLP**: Add layers via dialog, set units & activation.
  - **CNN/RNN**: Add Conv2D, pooling, Flatten, LSTM/GRU layers.
- Set training params: batch size, epochs, learning rate.
- Click **Train Neural Network**.
- Observe training/validation curves.

### Generative Adversarial Networks (GANs)

1. Load dataset.
2. Switch to **GAN** tab.
3. Set latent dimension, epochs, batch size.
4. Click **Train GAN**.
5. Monitor training log and progress bar.
6. Generated samples appear with PCA projection.

### Dimensionality Reduction

- **PCA/SVD**: Choose number of components, run, view explained variance plot.
- **t-SNE**: Select 2D/3D, run projection scatter plot.
- **LDA**: Select components, run histogram or scatter.

### Feature Extraction

- Interactive selection of PCA, SVD, t-SNE, LDA under **Feature Extraction** tab.

### Reinforcement Learning

- Under **Reinforcement Learning**, choose environment and algorithm.
- (Additional UI/controls under development.)

### Visualization & Metrics

- Raw data 3D scatter + histogram.
- Model predictions as 3D scatter against actuals.
- Metrics pane: Accuracy, Confusion Matrix for classification; MSE, RMSE, R² for regression.

---

## Logging & Troubleshooting

- Logs saved as `ml_gui_YYYYMMDD_HHMMSS.log` in working directory.
- GUI log window shows real-time entries.
- Check log file for full stack traces.
- Common fixes:
  - Verify dataset format for custom CSV.
  - Ensure selected features exist.
  - Confirm TensorFlow GPU dependencies if using GPU.

---

## Project Structure

```
├── 20067502.py       # Main application code
├── requirements.txt  # Dependency list
├── README.md         # This file
└── ml_gui_*.log      # Generated logs
```

---

## Contributing

Contributions welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit your changes and push: `git push origin feature/YourFeature`.
4. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
