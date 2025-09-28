# 🧠 Diabetes Prediction using ANN (PyTorch)

This project builds an **Artificial Neural Network (ANN)** using
**PyTorch** to predict whether a patient is diabetic based on medical
attributes such as glucose level, BMI, age, etc.\
It uses the **Pima Indians Diabetes dataset** and demonstrates **data
preprocessing, model training, evaluation, and prediction on new data**.

------------------------------------------------------------------------

## 📌 Problem Statement

Diabetes is a growing global health challenge, and early prediction can
help in timely medical intervention.\
This project aims to create a **touchless, automated, and reliable ML
model** to predict diabetes based on health metrics, supporting
**hygienic and efficient patient screening**.

------------------------------------------------------------------------

## 🚀 Features

-   ✅ **Data Preprocessing:** Cleanly loads and splits dataset into
    training/testing sets.\
-   ✅ **PyTorch ANN:** Implements a 3-layer fully connected neural
    network.\
-   ✅ **Model Training:** Uses **Adam optimizer** and
    **CrossEntropyLoss** for backpropagation.\
-   ✅ **Performance Evaluation:** Plots loss vs. epochs and visualizes
    confusion matrix with TP, TN, FP, FN values.\
-   ✅ **Model Persistence:** Saves and reloads trained model for future
    predictions.\
-   ✅ **New Data Prediction:** Allows real-time predictions for unseen
    data.

------------------------------------------------------------------------

## 🗂 Dataset

We use the **Pima Indians Diabetes Dataset** from [Plotly
Datasets](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv).\
**Features:** - Pregnancies\
- Glucose\
- BloodPressure\
- SkinThickness\
- Insulin\
- BMI\
- DiabetesPedigreeFunction\
- Age\
**Target:**\
- Outcome (0 → No Diabetes, 1 → Diabetes)

------------------------------------------------------------------------

## 📦 Installation & Setup

``` bash
# Clone the repository (optional if using local files)
git clone <your-repo-link>
cd diabetes-ann

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn torch

# Download the dataset
wget https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

# Run the script
python diabetes_ann.py
```

------------------------------------------------------------------------

## 🧠 Model Architecture

  Layer      Type            Units
  ---------- --------------- --------------
  Input      Linear          8 (features)
  Hidden-1   Linear + ReLU   50
  Hidden-2   Linear + ReLU   50
  Output     Linear          2 (classes)

------------------------------------------------------------------------

## 📊 Results & Evaluation

-   **Loss Function:** CrossEntropyLoss\
-   **Optimizer:** Adam\
-   **Epochs:** 500\
-   **Performance:** Achieved good accuracy on test set

Confusion Matrix Example:

                        Predicted Negative   Predicted Positive
  --------------------- -------------------- --------------------
  **Actual Negative**   TN                   FP
  **Actual Positive**   FN                   TP

Also prints: - ✅ True Positives (TP)\
- ✅ True Negatives (TN)\
- ✅ False Positives (FP)\
- ✅ False Negatives (FN)

------------------------------------------------------------------------

## 🔮 Predicting New Data

You can pass a **new patient's data** as a tensor and get a prediction:

``` python
list1 = [5.0, 140.0, 60.0, 35.0, 1.0, 34.6, 0.627, 40.0]
new_data = torch.tensor(list1)
with torch.no_grad():
    predicted_output = loaded_model(new_data)
    predicted_class = torch.argmax(predicted_output).item()
print(f"The predicted class is: {predicted_class}")
```

------------------------------------------------------------------------

## 📈 Visualization

-   **Epoch vs Loss Curve** (Training Progress)
-   **Confusion Matrix Heatmap** (Model Performance)

------------------------------------------------------------------------

## 🔮 Future Work

-   Add **hyperparameter tuning** (learning rate, hidden layers).\
-   Implement **dropout and batch normalization** to improve
    generalization.\
-   Deploy as **web app (Flask/FastAPI)** for real-world use.\
-   Integrate with **wearable devices** for real-time patient
    monitoring.
