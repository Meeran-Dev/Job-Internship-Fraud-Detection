# Job/Internship Scam Detector

## Project Description
This project is an ML application to predict whether a job or internship posting is likely to be a scam. It uses NLP techniques and classification models to analyze job details such as title, description, location, and requirements, and provides a confidence score for the prediction.

## Features
- User-friendly interface built with Streamlit for easy input of details.
- Predicts scam likelihood using a pre-trained machine learning model.
- Displays confidence scores for predictions.
- Custom evaluation metric combining recall and precision for model performance.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Job-Internship-Fraud-Detection
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Ensure the dataset `fake_job_postings.csv` is in the project directory.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Enter the job or internship details in the provided fields:
   - Title
   - Description
   - Location
   - Requirements (optional)

3. Click "Predict" to see whether the posting is a scam or not.

## Model Training and Evaluation

- The model is trained using a public dataset taken from Kaggle - [Real / Fake Job Posting](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).
- Text features from title, location, description, and requirements are combined and vectorized using TF-IDF.
- Synthetic Minority Over-sampling Technique (SMOTE) is applied to handle class imbalance.
- Multiple classifiers were evaluated including Random Forest, XGBoost, and Naive Bayes.
- Hyperparameter tuning was performed using GridSearchCV.
- A custom weighted recall-precision metric was used to select the best model.
- The final model is saved as `scam_prediction_model.pkl` and loaded by the Streamlit app.

- Final Classification Report
              precision    recall  f1-score   support

           0       0.99      0.95      0.97      4212
           1       0.39      0.79      0.52       185

    accuracy                           0.94      4397
   macro avg       0.69      0.87      0.74      4397
weighted avg       0.96      0.94      0.95      4397

