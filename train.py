import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='/opt/ml/input/data/train/')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model/')
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(os.path.join(args.train_data, 'diabetes.csv'))

    # Handle duplicates
    df = df.drop_duplicates()
    df = df[df['gender'] != 'Other']

    # Recategorize smoking history
    def recategorize_smoking(smoking_status):
        if smoking_status in ['never', 'No Info']:
            return 'non-smoker'
        elif smoking_status == 'current':
            return 'current'
        elif smoking_status in ['ever', 'former', 'not current']:
            return 'past_smoker'

    df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)

    # Split features and target
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']),
            ('cat', OneHotEncoder(), ['gender', 'smoking_history'])
        ]
    )

    # Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    grid_search.fit(X_train, y_train)

    # Save the model
    import joblib
    joblib.dump(grid_search.best_estimator_, os.path.join(args.model_dir, 'model.joblib'))

    # Evaluate the model
    y_pred = grid_search.predict(X_test)
    print("Model Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()
