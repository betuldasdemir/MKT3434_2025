import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        main_layout = QVBoxLayout()
        
        # Upper section for data loading and preprocessing
        upper_layout = QHBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMaximumHeight(25)  # Set maximum height
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.setMaximumHeight(25)  # Set maximum height
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to upper layout
        upper_layout.addWidget(QLabel("Dataset:"))
        upper_layout.addWidget(self.dataset_combo)
        upper_layout.addWidget(self.load_btn)
        upper_layout.addWidget(QLabel("Scaling:"))
        upper_layout.addWidget(self.scaling_combo)
        upper_layout.addWidget(QLabel("Test Split:"))
        upper_layout.addWidget(self.split_spin)
        
        # Middle section for SVM configuration
        svm_group = QGroupBox("SVM Configuration")
        svm_layout = QGridLayout()
        
        # Kernel selection
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems([
            "linear",
            "rbf",
            "poly",
            "sigmoid"
        ])
        svm_layout.addWidget(QLabel("Kernel:"), 0, 0)
        svm_layout.addWidget(self.kernel_combo, 0, 1)
        
        # C parameter (regularization)
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.01, 100.0)
        self.c_spin.setValue(1.0)
        self.c_spin.setSingleStep(0.1)
        self.c_spin.setDecimals(2)
        svm_layout.addWidget(QLabel("C (Regularization):"), 0, 2)
        svm_layout.addWidget(self.c_spin, 0, 3)
        
        # Gamma parameter for RBF, poly and sigmoid
        self.gamma_combo = QComboBox()
        self.gamma_combo.addItems([
            "scale",
            "auto",
            "custom"
        ])
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0001, 100.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setDecimals(4)
        self.gamma_spin.setEnabled(False)
        self.gamma_combo.currentTextChanged.connect(
            lambda x: self.gamma_spin.setEnabled(x == "custom"))
        svm_layout.addWidget(QLabel("Gamma:"), 1, 0)
        svm_layout.addWidget(self.gamma_combo, 1, 1)
        svm_layout.addWidget(self.gamma_spin, 1, 2)
        
        # Degree for polynomial kernel
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(2, 10)
        self.degree_spin.setValue(3)
        self.degree_spin.setEnabled(False)
        self.kernel_combo.currentTextChanged.connect(
            lambda x: self.degree_spin.setEnabled(x == "poly"))
        svm_layout.addWidget(QLabel("Polynomial Degree:"), 1, 3)
        svm_layout.addWidget(self.degree_spin, 1, 4)
        
        # Epsilon for SVR
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.01, 1.0)
        self.epsilon_spin.setValue(0.1)
        self.epsilon_spin.setSingleStep(0.01)
        self.epsilon_spin.setDecimals(3)
        svm_layout.addWidget(QLabel("Epsilon (SVR):"), 2, 0)
        svm_layout.addWidget(self.epsilon_spin, 2, 1)
        
        svm_group.setLayout(svm_layout)
        
        # Loss functions section
        loss_group = QGroupBox("Loss Functions")
        loss_layout = QGridLayout()
        
        # Regression loss options
        reg_group = QGroupBox("Regression Losses")
        reg_layout = QVBoxLayout()
        self.reg_loss_combo = QComboBox()
        self.reg_loss_combo.addItems([
            "Mean Squared Error (MSE)",
            "Mean Absolute Error (MAE)",
            "Root Mean Squared Error (RMSE)",
            "Huber Loss"
        ])
        
        # Huber delta parameter
        self.huber_delta_spin = QDoubleSpinBox()
        self.huber_delta_spin.setRange(0.1, 5.0)
        self.huber_delta_spin.setValue(1.0)
        self.huber_delta_spin.setSingleStep(0.1)
        self.huber_delta_spin.setEnabled(False)
        self.reg_loss_combo.currentTextChanged.connect(
            lambda x: self.huber_delta_spin.setEnabled(x == "Huber Loss"))
        
        reg_layout.addWidget(self.reg_loss_combo)
        reg_layout.addWidget(QLabel("Huber Delta:"))
        reg_layout.addWidget(self.huber_delta_spin)
        reg_group.setLayout(reg_layout)
        
        # Classification loss options
        class_group = QGroupBox("Classification Losses")
        class_layout = QVBoxLayout()
        self.class_loss_combo = QComboBox()
        self.class_loss_combo.addItems([
            "Binary Cross Entropy",
            "Categorical Cross Entropy",
            "Sparse Categorical Cross Entropy",
            "Hinge Loss"
        ])
        class_layout.addWidget(self.class_loss_combo)
        class_group.setLayout(class_layout)
        
        # Add loss groups to loss layout
        loss_layout.addWidget(reg_group, 0, 0)
        loss_layout.addWidget(class_group, 0, 1)
        loss_group.setLayout(loss_layout)
        
        # Add all sections to main layout
        main_layout.addLayout(upper_layout)
        main_layout.addWidget(svm_group)
        main_layout.addWidget(loss_group)
        
        data_group.setLayout(main_layout)
        self.layout.addWidget(data_group)

    def calculate_loss(self, y_true, y_pred):
        """Calculate loss based on selected loss function"""
        try:
            if self.is_classification_task():
                # Convert y_true to appropriate type for classification
                y_true = tf.cast(y_true, tf.int32)
                
                loss_name = self.class_loss_combo.currentText()
                if loss_name == "Binary Cross Entropy":
                    return tf.keras.losses.binary_crossentropy(y_true, y_pred).numpy().mean()
                elif loss_name == "Categorical Cross Entropy":
                    return tf.keras.losses.categorical_crossentropy(y_true, y_pred).numpy().mean()
                elif loss_name == "Sparse Categorical Cross Entropy":
                    # Convert to integers for sparse categorical crossentropy
                    y_true_int = tf.cast(tf.round(y_true), tf.int32)
                    return tf.keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred).numpy().mean()
                elif loss_name == "Hinge Loss":
                    return tf.keras.losses.hinge(y_true, y_pred).numpy().mean()
            
            return 0.0
        except Exception as e:
            self.show_error(f"Error calculating loss: {str(e)}")
            return 0.0
            
    def get_svm_params(self):
        """Get the current SVM parameters"""
        params = {
            'kernel': self.kernel_combo.currentText(),
            'C': self.c_spin.value()
        }
        
        # Add gamma parameter
        if params['kernel'] in ['rbf', 'poly', 'sigmoid']:
            if self.gamma_combo.currentText() == 'custom':
                params['gamma'] = self.gamma_spin.value()
            else:
                params['gamma'] = self.gamma_combo.currentText()
        
        # Add degree parameter for polynomial kernel
        if params['kernel'] == 'poly':
            params['degree'] = self.degree_spin.value()
        
        return params

    def create_svm_model(self, is_classification=True):
        """Create SVM model with current parameters"""
        params = self.get_svm_params()
        
        if is_classification:
            model = SVC(**params)
        else:
            params['epsilon'] = self.epsilon_spin.value()
            model = SVR(**params)
        
        return model
        
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def select_target_column(self, columns):
        """Dialog to select target column"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create left and right plot sections
        left_plot_group = QGroupBox("Input Data")
        right_plot_group = QGroupBox("Model Prediction")
        
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # Create matplotlib figures
        self.input_figure = Figure(figsize=(8, 7))
        self.prediction_figure = Figure(figsize=(8, 7))
        self.input_canvas = FigureCanvas(self.input_figure)
        self.prediction_canvas = FigureCanvas(self.prediction_figure)
        
        left_layout.addWidget(self.input_canvas)
        right_layout.addWidget(self.prediction_canvas)
        
        left_plot_group.setLayout(left_layout)
        right_plot_group.setLayout(right_layout)
        
        # Add plots to main layout
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(left_plot_group)
        plot_layout.addWidget(right_plot_group)
        
        # Metrics display
        metrics_group = QGroupBox("Model Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        
        # Combine all layouts
        main_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout)
        main_layout.addWidget(metrics_group)
        
        viz_group.setLayout(main_layout)
        self.layout.addWidget(viz_group)

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
                X, y = data.data, data.target
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
                X, y = data.data, data.target
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
                X, y = data.data, data.target
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
                X, y = data.data, data.target
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                # Flatten images
                self.X_train = X_train.reshape((X_train.shape[0], -1))
                self.X_test = X_test.reshape((X_test.shape[0], -1))
                self.y_train, self.y_test = y_train, y_test
                self.apply_scaling()
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                self.plot_raw_data()
                return
            
            # Split data for non-MNIST datasets
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Apply scaling if selected
            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            self.plot_raw_data()
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1).values
                    y = data[target_col].values
                    
                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    self.plot_raw_data()
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def plot_raw_data(self):
        """Plot the input data"""
        try:
            if self.X_train is None or self.y_train is None:
                return
                
            self.input_figure.clear()
            ax = self.input_figure.add_subplot(111)
            
            # For 2D data, plot scatter
            if self.X_train.shape[1] == 2:
                scatter = ax.scatter(self.X_train[:, 0], self.X_train[:, 1], 
                                  c=self.y_train, cmap='viridis')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                self.input_figure.colorbar(scatter)
            # For higher dimensions, use PCA to visualize
            else:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(self.X_train)
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=self.y_train, cmap='viridis')
                ax.set_xlabel('First Principal Component')
                ax.set_ylabel('Second Principal Component')
                self.input_figure.colorbar(scatter)
            
            ax.set_title('Training Data')
            self.input_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error plotting data: {str(e)}")

    def train_model(self, name, param_widgets):
        """Train the selected model with parameters"""
        try:
            model = None
            if name == "Linear Regression":
                model = LinearRegression(
                    fit_intercept=param_widgets['fit_intercept'].isChecked(),
                    normalize=param_widgets['normalize'].isChecked()
                )
            elif name == "Logistic Regression":
                model = LogisticRegression(
                    C=param_widgets['C'].value(),
                    max_iter=param_widgets['max_iter'].value(),
                    multi_class=param_widgets['multi_class'].currentText()
                )
            elif name == "Naive Bayes":
                model = GaussianNB(
                    var_smoothing=param_widgets['var_smoothing'].value()
                )
            elif name == "Support Vector Machine":
                model = SVC(
                    C=param_widgets['C'].value(),
                    kernel=param_widgets['kernel'].currentText(),
                    degree=param_widgets['degree'].value()
                )
            elif name == "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=param_widgets['max_depth'].value(),
                    min_samples_split=param_widgets['min_samples_split'].value(),
                    criterion=param_widgets['criterion'].currentText()
                )
            elif name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=param_widgets['n_estimators'].value(),
                    max_depth=param_widgets['max_depth'].value(),
                    min_samples_split=param_widgets['min_samples_split'].value()
                )
            elif name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(
                    n_neighbors=param_widgets['n_neighbors'].value(),
                    weights=param_widgets['weights'].currentText(),
                    metric=param_widgets['metric'].currentText()
                )
            elif name == "K-Means Clustering":
                model = KMeans(
                    n_clusters=param_widgets['n_clusters'].value(),
                    max_iter=param_widgets['max_iter'].value(),
                    n_init=param_widgets['n_init'].value()
                )
                model.fit(np.concatenate((self.X_train, self.X_test)))
                y_pred = model.predict(self.X_test)
                self.update_visualization(y_pred)
                self.update_metrics(y_pred)
                return
            elif name == "Principal Component Analysis":
                pca = PCA(
                    n_components=param_widgets['n_components'].value(),
                    whiten=param_widgets['whiten'].isChecked()
                )
                self.X_train = pca.fit_transform(self.X_train)
                self.X_test = pca.transform(self.X_test)
                self.plot_raw_data()
                return

            if model is None:
                self.show_error("Model not implemented")
                return

            model.fit(self.X_train, self.y_train)
            self.current_model = model
            y_pred = model.predict(self.X_test)
            
            self.plot_model_prediction()
            self.update_metrics(y_pred)
            self.status_bar.showMessage(f"{name} training complete")
            
        except Exception as e:
            self.show_error(f"Error training {name}: {str(e)}")

    def plot_model_prediction(self):
        """Plot the model predictions"""
        try:
            if self.current_model is None:
                return
                
            self.prediction_figure.clear()
            ax = self.prediction_figure.add_subplot(111)
            
            # For 2D data, plot decision boundary
            if self.X_train.shape[1] == 2:
                # Create mesh grid
                x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
                y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                   np.arange(y_min, y_max, 0.02))
                
                # Make predictions on mesh grid
                Z = self.current_model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary and data points
                ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1], 
                                  c=self.y_test, cmap='viridis', alpha=0.8)
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                self.prediction_figure.colorbar(scatter)
            else:
                # For higher dimensions, use PCA visualization with predictions
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(self.X_test)
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=self.current_model.predict(self.X_test),
                                  cmap='viridis')
                ax.set_xlabel('First Principal Component')
                ax.set_ylabel('Second Principal Component')
                self.prediction_figure.colorbar(scatter)
            
            ax.set_title('Model Predictions')
            self.prediction_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error plotting predictions: {str(e)}")

    def update_metrics(self, y_pred):
        """Update metrics display"""
        try:
            if self.y_test is not None and y_pred is not None:
                metrics_text = ""
                
                # Calculate and display loss
                loss = self.calculate_loss(self.y_test, y_pred)
                metrics_text += f"Selected Loss Function: {self.get_selected_loss()}\n"
                metrics_text += f"Loss Value: {loss:.4f}\n\n"
                
                # Add additional metrics based on task type
                if self.is_classification_task():
                    accuracy = accuracy_score(self.y_test, y_pred)
                    conf_matrix = confusion_matrix(self.y_test, y_pred)
                    metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
                    metrics_text += "Confusion Matrix:\n"
                    metrics_text += str(conf_matrix)
                else:
                    mse = mean_squared_error(self.y_test, y_pred)
                    r2 = self.current_model.score(self.X_test, self.y_test)
                    
                    # Only show MSE if it's not already shown as the selected loss
                    if self.get_selected_loss() != "Mean Squared Error (MSE)":
                        metrics_text += f"Mean Squared Error: {mse:.4f}\n"
                    
                    metrics_text += f"R² Score: {r2:.4f}"
                
                # Set the metrics text with fixed-width font for better alignment
                self.metrics_text.setFontFamily("Courier")
                self.metrics_text.setText(metrics_text)
                
        except Exception as e:
            self.show_error(f"Error updating metrics: {str(e)}")

    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Naive Bayes", self.create_naive_bayes_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox"}
        )
        regression_layout.addWidget(lr_group)
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double"}
        )
        classification_layout.addWidget(nb_group)
        
        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int"}
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group

    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        try:
            # Get parameters
            epochs = self.epochs_spin.value()
            batch_size = self.batch_size_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Get the selected loss function
            loss_function = None
            if self.is_classification_task():
                loss_name = self.class_loss_combo.currentText()
                if loss_name == "Binary Cross Entropy":
                    loss_function = 'binary_crossentropy'
                elif loss_name == "Categorical Cross Entropy":
                    loss_function = 'categorical_crossentropy'
                elif loss_name == "Sparse Categorical Cross Entropy":
                    # Convert to integers for sparse categorical crossentropy
                    y_true_int = tf.cast(tf.round(self.y_train), tf.int32)
                    loss_function = 'sparse_categorical_crossentropy'
                elif loss_name == "Hinge Loss":
                    loss_function = 'hinge'
            else:
                loss_name = self.reg_loss_combo.currentText()
                if loss_name == "Mean Squared Error (MSE)":
                    loss_function = 'mse'
                elif loss_name == "Mean Absolute Error (MAE)":
                    loss_function = 'mae'
                elif loss_name == "Root Mean Squared Error (RMSE)":
                    # TF doesn't have direct RMSE, use MSE and take sqrt later
                    loss_function = 'mse'
                elif loss_name == "Huber Loss":
                    delta = self.huber_delta_spin.value()
                    loss_function = tf.keras.losses.Huber(delta=delta)
            
            # If no specific loss function is selected, use defaults
            if loss_function is None:
                loss_function = 'categorical_crossentropy' if self.is_classification_task() else 'mse'
            
            # Create model
            model = models.Sequential()
            
            # Add layers from configuration
            input_shape = (self.X_train.shape[1],)
            X_train_model = self.X_train
            X_test_model = self.X_test
            
            # Check if convolutional layers are present
            is_conv = any('Conv' in layer['type'] for layer in self.layer_config)
            
            if is_conv:
                # Assuming square images
                img_size = int(np.sqrt(self.X_train.shape[1]))
                input_shape = (img_size, img_size, 1)
                X_train_model = self.X_train.reshape(-1, img_size, img_size, 1)
                X_test_model = self.X_test.reshape(-1, img_size, img_size, 1)
            
            # Add input layer
            first_layer = self.layer_config[0]
            if first_layer['type'] == 'Dense':
                model.add(layers.Dense(first_layer['params']['units'], 
                                      activation=first_layer['params']['activation'],
                                      input_shape=input_shape))
            elif first_layer['type'] == 'Conv2D':
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params = first_layer['params'].copy()
                    params['input_shape'] = input_shape
                    model.add(layers.Conv2D(**params))
                else:
                    model.add(layers.Conv2D(**first_layer['params']))
            
            # Add hidden layers
            for layer_info in self.layer_config[1:]:
                if layer_info['type'] == 'Dense':
                    model.add(layers.Dense(layer_info['params']['units'], 
                                         activation=layer_info['params']['activation']))
                elif layer_info['type'] == 'Conv2D':
                    model.add(layers.Conv2D(**layer_info['params']))
                elif layer_info['type'] == 'MaxPooling2D':
                    model.add(layers.MaxPooling2D(pool_size=layer_info['params']['pool_size']))
                elif layer_info['type'] == 'Flatten':
                    model.add(layers.Flatten())
                elif layer_info['type'] == 'Dropout':
                    model.add(layers.Dropout(layer_info['params']['rate']))
            
            # Add output layer based on task type
            if self.is_classification_task():
                n_classes = len(np.unique(self.y_train))
                if n_classes == 2:
                    model.add(layers.Dense(1, activation='sigmoid'))
                else:
                    model.add(layers.Dense(n_classes, activation='softmax'))
            else:
                model.add(layers.Dense(1, activation='linear'))
            
            # Compile model with selected loss function
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            
            # Create a custom loss function wrapper for sparse categorical crossentropy
            # to ensure proper type conversion
            if self.is_classification_task() and isinstance(loss_function, str) and loss_function == 'sparse_categorical_crossentropy':
                def safe_sparse_categorical_crossentropy(y_true, y_pred):
                    y_true_int = tf.cast(tf.round(y_true), tf.int32)
                    return tf.keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred)
                model.compile(optimizer=optimizer,
                            loss=safe_sparse_categorical_crossentropy,
                            metrics=['accuracy'])
            else:
                model.compile(optimizer=optimizer,
                            loss=loss_function,
                            metrics=['accuracy'] if self.is_classification_task() else ['mse'])
            
            # Train model
            self.update_status("Training deep learning model...")
            
            # Check if using sparse categorical crossentropy and convert labels to integers if needed
            y_train_model = self.y_train
            y_test_model = self.y_test
            
            if self.is_classification_task() and isinstance(loss_function, str) and loss_function == 'sparse_categorical_crossentropy':
                # Convert labels to integers for sparse categorical crossentropy
                y_train_model = tf.cast(tf.round(self.y_train), tf.int32)
                y_test_model = tf.cast(tf.round(self.y_test), tf.int32)
            
            history = model.fit(
                X_train_model, y_train_model,
                validation_data=(X_test_model, y_test_model),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            # Store model
            self.current_model = model
            
            # Make predictions
            y_pred = model.predict(X_test_model)
            
            # Convert predictions to appropriate format
            if self.is_classification_task():
                n_classes = len(np.unique(self.y_train))
                if n_classes == 2:
                    y_pred = (y_pred > 0.5).astype(int).flatten()
                else:
                    y_pred = np.argmax(y_pred, axis=1)
            
            # Update metrics and plot
            self.update_metrics(y_pred)
            self.plot_model_predictions()
            
            self.update_status("Deep learning model training completed")
            
        except Exception as e:
            self.show_error(f"Error training deep learning model: {str(e)}")

    def create_naive_bayes_tab(self):
        """Create the Naive Bayes tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Gaussian Naive Bayes configuration
        nb_group = QGroupBox("Gaussian Naive Bayes Configuration")
        nb_layout = QGridLayout()
        
        # Var smoothing parameter
        self.var_smoothing_spin = QDoubleSpinBox()
        self.var_smoothing_spin.setRange(1e-12, 1.0)
        self.var_smoothing_spin.setValue(1e-9)  # Default scikit-learn value
        self.var_smoothing_spin.setDecimals(12)
        self.var_smoothing_spin.setSingleStep(1e-10)
        nb_layout.addWidget(QLabel("Var Smoothing:"), 0, 0)
        nb_layout.addWidget(self.var_smoothing_spin, 0, 1)
        
        # Prior probabilities
        self.prior_combo = QComboBox()
        self.prior_combo.addItems(["Uniform", "Custom"])
        nb_layout.addWidget(QLabel("Prior Probabilities:"), 0, 2)
        nb_layout.addWidget(self.prior_combo, 0, 3)
        
        # Custom priors entry (enabled only when custom is selected)
        self.prior_entry = QLineEdit()
        self.prior_entry.setPlaceholderText("e.g., 0.3,0.7 for binary classification")
        self.prior_entry.setEnabled(False)
        self.prior_combo.currentTextChanged.connect(
            lambda x: self.prior_entry.setEnabled(x == "Custom"))
        nb_layout.addWidget(QLabel("Custom Priors:"), 1, 0)
        nb_layout.addWidget(self.prior_entry, 1, 1, 1, 3)
        
        # Explanation for var_smoothing
        explanation = QLabel("Var smoothing: Portion of the largest variance of all features added to variances for stability (scikit-learn default: 1e-9)")
        explanation.setWordWrap(True)
        nb_layout.addWidget(explanation, 2, 0, 1, 4)
        
        # Button to train the model
        self.train_nb_btn = QPushButton("Train Gaussian Naive Bayes")
        self.train_nb_btn.clicked.connect(self.train_naive_bayes)
        nb_layout.addWidget(self.train_nb_btn, 3, 0, 1, 4)
        
        nb_group.setLayout(nb_layout)
        layout.addWidget(nb_group)
        
        # Training progress and results
        layout.addWidget(self.create_progress_group())
        
        return widget

    def create_progress_group(self):
        """Create a group for displaying training progress and results"""
        group = QGroupBox("Training Progress and Results")
        layout = QVBoxLayout()
        
        # Progress bar
        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)
        
        # Metrics text display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumHeight(100)
        layout.addWidget(self.metrics_text)
        
        # Visualization
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        group.setLayout(layout)
        return group

    def train_naive_bayes(self):
        """Train Naive Bayes model with current parameters"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load data first")
            return
            
        try:
            self.update_status("Training Gaussian Naive Bayes model...")
            
            # Get parameters from UI
            var_smoothing = self.var_smoothing_spin.value()
            
            # Handle priors
            priors = None
            if self.prior_combo.currentText() == "Custom":
                try:
                    priors_text = self.prior_entry.text()
                    if priors_text:
                        priors = np.array([float(x) for x in priors_text.split(',')])
                        # Verify priors sum to 1
                        if abs(np.sum(priors) - 1.0) > 1e-10:
                            self.show_error("Custom priors must sum to 1.0")
                            return
                except Exception as e:
                    self.show_error(f"Invalid custom priors: {str(e)}")
                    return
            
            # Create and train the model
            self.current_model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
            self.current_model.fit(self.X_train, self.y_train)
            
            # Make predictions and update visualization
            y_pred = self.current_model.predict(self.X_test)
            self.update_metrics(y_pred)
            self.plot_model_predictions()
            
            self.update_status("Gaussian Naive Bayes model training completed")
            
        except Exception as e:
            self.show_error(f"Error training Naive Bayes model: {str(e)}")

    def get_selected_loss(self):
        """Get the selected loss function based on the problem type"""
        if self.is_classification_task():
            return self.class_loss_combo.currentText()
        else:
            return self.reg_loss_combo.currentText()

    def is_classification_task(self):
        """Determine if the current task is classification or regression"""
        if self.y_train is not None:
            return len(np.unique(self.y_train)) <= 10
        return False

    def plot_model_predictions(self):
        """Plot the model predictions"""
        try:
            if self.current_model is None:
                return
                
            self.prediction_figure.clear()
            ax = self.prediction_figure.add_subplot(111)
            
            # For 2D data, plot decision boundary
            if self.X_train.shape[1] == 2:
                # Create mesh grid
                x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
                y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                   np.arange(y_min, y_max, 0.02))
                
                # Make predictions on mesh grid
                Z = self.current_model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary and data points
                ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1], 
                                  c=self.y_test, cmap='viridis', alpha=0.8)
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                self.prediction_figure.colorbar(scatter)
            else:
                # For higher dimensions, use PCA visualization with predictions
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(self.X_test)
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=self.current_model.predict(self.X_test),
                                  cmap='viridis')
                ax.set_xlabel('First Principal Component')
                ax.set_ylabel('Second Principal Component')
                self.prediction_figure.colorbar(scatter)
            
            ax.set_title('Model Predictions')
            self.prediction_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error plotting predictions: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
