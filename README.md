Credit Card Fraud Detection using Machine Learning and ANN
This project implements a credit card fraud detection system using machine learning and an artificial neural network (ANN). It focuses on classifying transactions as fraudulent or non-fraudulent based on the given dataset.

Table of Contents
Overview
Features
Technologies Used
Dataset
Installation
Usage
Results
Contributing
License
Overview
This project aims to analyze and detect fraudulent credit card transactions using a combination of statistical analysis, machine learning models, and an artificial neural network. The dataset used contains transaction details such as time, amount, and a Class column indicating whether the transaction is fraudulent (1) or non-fraudulent (0).

Features
Data exploration and visualization to understand class distributions and correlations.
Data preprocessing, including scaling and handling class imbalances.
Implementation of an artificial neural network (ANN) for classification.
Evaluation of the model's performance using precision, recall, F1-score, and other metrics.
Technologies Used
Python: Programming language.
Pandas, NumPy: Data manipulation and analysis.
Matplotlib, Seaborn: Data visualization.
Scikit-learn: Data preprocessing and evaluation metrics.
TensorFlow (Keras): ANN model development.
Dataset
The dataset used is ProjectCreditCard.xlsx. Ensure the file is present in the root directory for the code to execute successfully. It includes the following features:

Time: Seconds elapsed between transactions.
Amount: Transaction amount.
Class: Target variable (1 = Fraudulent, 0 = Non-Fraudulent).
Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install required packages:
bash
Copy
Edit
pip install -r requirements.txt
Ensure the dataset file (ProjectCreditCard.xlsx) is in the project directory.
Usage
Run the script:
bash
Copy
Edit
python fraud_detection.py
Modify the script to experiment with the dataset, hyperparameters, or model structure.
Results
Time consumed for training: 1260.5851862430573 seconds
Model saved successfully!
2671/2671 ━━━━━━━━━━━━━━━━━━━━ 8s 3ms/step - loss: 0.0251 - precision: 0.7268 - recall: 0.0148
Model evaluation: [0.024628935381770134, 1.0, 0.022058824077248573]

Reloading the model for validation...
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
C:\Users\DELL\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
2671/2671 ━━━━━━━━━━━━━━━━━━━━ 6s 2ms/step
Model Accuracy: 0.9984
Confusion Matrix:
 [[85307     0]
 [  134     2]]
Classification Report:
               precision    recall  f1-score   support

      Normal       1.00      1.00      1.00     85307
       Fraud       1.00      0.01      0.03       136

    accuracy                           1.00     85443
   macro avg       1.00      0.51      0.51     85443
weighted avg       1.00      1.00      1.00     85443


Contributing
Contributions are welcome! Feel free to raise issues or submit pull requests to improve this project.



