# Regression Model for Tabular Data

This project involves building a regression model to predict a target value from a dataset with anonymized features. The goal is to minimize the RMSE (Root Mean Squared Error) on the provided test dataset.

## Task

Regression on the tabular data. General Machine Learning

You have a dataset (train.csv) that contains 53 anonymized features and a target
column. Your task is to build a model that predicts a target based on the proposed
features. Please provide predictions for the hidden_test.csv file. Target metric is RMSE.

The main goal is to provide github repository that contains:

● jupyter notebook with exploratory data analysis;
● train.py python script for model training;
● predict.py python script for model inference on test data;
● file with prediction results;
● readme file that contains instructions about project setup and general guidance
around project;
● requirements.txt file.

Please provide documented code. Scripts (train.py and predict.py) should be able
to be executed from the terminal.

## Project Structure

1. **`train.py`**: Python script for training the regression model.
2. **`predict.py`**: Python script for making predictions on the test data.
3. **`EDA.ipynb`**: Jupyter notebook containing exploratory data analysis (EDA).
4. **`predictions.csv`**: File with prediction results.
5. **`requirements.txt`**: List of Python packages required for the project.
6. **`README.md`**: This file, providing general project information and setup instructions.

## Setup Instructions

### Prerequisites

Make sure you have Python 3.10 or later installed on your system.

### Installing Dependencies

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_name>

2. Create and activate a virtual environment:

   ```bash
   python3.10 -m venv myenv
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`

3. Install the required packages:

   ```bash
   pip install -r requirements.txt

### Running the code

1. Train the model:

   ```bash
   python train.py

2. Make predictions:

   ```bash
   python predict.py