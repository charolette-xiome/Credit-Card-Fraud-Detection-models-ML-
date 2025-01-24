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
The ANN model was trained for 300 epochs using binary cross-entropy loss and Adam optimizer.
Key evaluation metrics:
Precision: Indicates how many identified frauds were actual frauds.
Recall: Shows the percentage of actual frauds identified.
F1-Score: Balance between precision and recall.
Graphs for loss, precision, and recall over epochs are plotted for better understanding.

Contributing
Contributions are welcome! Feel free to raise issues or submit pull requests to improve this project.
