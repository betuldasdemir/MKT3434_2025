# README - Enhanced GUI for Machine Learning Models

## Project Overview

This project enhances a provided GUI application by integrating multiple machine learning models and improving its functionality. The updated version supports regression and classification models with various loss functions, kernel selections, and missing data handling methods.

---

## Features Implemented

### 1. Regression Models

- Implemented Linear Regression, Decision Tree Regression, and Support Vector Regression (SVR).
- Added Mean Squared Error (MSE), Mean Absolute Error (MAE), and Huber Loss options.
- Enabled kernel selection (Linear, RBF, Polynomial) and hyperparameter tuning (C, Epsilon) for SVR.
- Tested SVR on the Boston Housing dataset.

### 2. Classification Models

- Implemented Logistic Regression, Decision Tree Classification, and Support Vector Machines (SVM).
- Added Cross-Entropy and Hinge Loss options.
- Integrated Gaussian Naïve Bayes (GaussianNB) with adjustable var_smoothing and prior probabilities (uniform/user-defined).
- Added input fields for Bayesian prior probabilities (e.g., [0.3, 0.7]).

### 3. Missing Data Handling

- Added options to handle missing values using Mean Imputation, Interpolation, and Forward/Backward Fill.
- Implemented logic to apply selected methods before splitting data into training/testing sets.
- Utilized Scikit-learn's `SimpleImputer` for efficient data preprocessing.

### 4. GUI Enhancements

- Updated the GUI to dynamically update the model training logic based on selected loss functions.
- Added a new section for Bayesian classification.
- Improved visualization panel to display results effectively.

### 5. Visualization

- Implemented 3D scatter plot for raw data visualization.
- Added histograms to analyze data distributions.
- Enabled selection of X, Y, and Z features for visualization.
- Displayed model predictions and actual values in a 3D graph.
- Showed performance metrics (error, accuracy, confusion matrix, etc.) in a text box.

### 6. Model Training

- Supported classic machine learning algorithms: Linear Regression, Logistic Regression, Naïve Bayes, SVM, Decision Tree, Random Forest, K-Nearest Neighbors.
- Allowed users to set parameters for each algorithm.
- Enabled selection of loss functions: Cross Entropy, Binary Cross Entropy, Hinge Loss (classification), and MSE, MAE, Huber Loss (regression).
- Automatically updated model predictions and performance metrics upon training completion.

### 7. Deep Learning Section

- Configurable Multi-Layer Perceptron (MLP), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN).
- Dynamic layer addition: Dense, Conv2D, MaxPooling2D, Flatten, Dropout.
- Training parameters: Batch size, epochs, learning rate.
- Progress tracking and visualization of training history.

### 8. Additional Features

- Tabbed layout for easy navigation between Machine Learning, Deep Learning, Dimensionality Reduction, and Reinforcement Learning sections.
- Error and warning message handling.
- Compact and user-friendly interface.

---

## Installation

### Requirements

- Python 3.7+
- Required Python packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `PyQt6`
  - `tensorflow`

### Installation Steps

1. Ensure Python and pip are installed on your system.
2. Install the required packages using the following command:
   ```bash
   pip install numpy pandas scikit-learn matplotlib PyQt6 tensorflow
   ```

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/bayraktare/MKT3434_2025.git
   ```
2. Checkout to your branch:
   ```bash
   git checkout -b <StudentID>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python <StudentID.py>
   ```
5. Select a model, configure its hyperparameters, and choose a loss function.
6. Handle missing data using the provided options before training models.
7. Train and evaluate models with visualized results.

---

## Project Structure

- `main.py`: Main application script.
- `gui/`: Contains GUI components.
- `models/`: Contains implemented machine learning models.
- `utils/`: Utility functions for data processing and visualization.

---

## Troubleshooting

If you encounter any issues:
- Ensure all dependencies are installed correctly.
- Check for any missing data in your dataset before training.
- Verify that your Python version is compatible.
