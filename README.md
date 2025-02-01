# Diabetes Prediction

## Overview
This project implements a machine learning model for predicting diabetes using a dataset. The model is built using TensorFlow's Keras library and a simple neural network. It also utilizes Scikit-Learn for data preprocessing and evaluation.

## Dataset
The dataset used in this project is assumed to be the **diabetes.csv** file. It contains features related to patient health metrics and a label indicating the presence or absence of diabetes.

## Requirements
Ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Implementation Details

### Steps Involved
1. **Load Dataset**: Reads the diabetes dataset using Pandas.
2. **Data Preprocessing**:
   - Checks for missing values.
   - Splits data into features (`X`) and labels (`Y`).
   - Splits data into training and testing sets (80-20 split).
   - Standardizes the features using `StandardScaler`.
3. **Model Creation**:
   - Defines a Sequential Neural Network with three layers:
     - Input layer with 10 neurons and ReLU activation.
     - Hidden layer with 5 neurons and ReLU activation.
     - Output layer with 1 neuron and sigmoid activation.
   - Compiles the model using the Adam optimizer and binary cross-entropy loss function.
4. **Training & Evaluation**:
   - Trains the model for 100 epochs.
   - Plots the loss graph over epochs.

## Code Structure
- **Data Preprocessing**: Loads and standardizes the dataset.
- **Model Building**: Defines and compiles the neural network.
- **Training & Visualization**: Trains the model and plots the loss graph.

## Usage
Run the script using Python:
```bash
python diabetes_prediction.py
```
Ensure the `diabetes.csv` file is available in the working directory.

## Results
The trained model predicts diabetes based on given health metrics. The performance can be evaluated using accuracy metrics.

## Future Enhancements
- Improve model accuracy by tuning hyperparameters.
- Use different architectures (CNNs, RNNs) for better performance.
- Implement a GUI for easy user interaction.


