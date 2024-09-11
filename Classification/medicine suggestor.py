import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

medical_data = pd.read_csv('medical data.csv')
medical_data.dropna(subset=['Symptoms', 'Causes'], inplace=True)
medical_data['Symptom_Cause'] = medical_data['Symptoms'] + ' ' + medical_data['Causes']
X = medical_data['Symptom_Cause']
y = medical_data['Medicine']
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto'],
    'svm__kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y_encoded)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")

best_model = grid_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")


def suggest_medicine(symptoms, causes):
    symptom_cause = symptoms + ' ' + causes
    predicted_medicine_encoded = best_model.predict([symptom_cause])
    predicted_medicine = label_encoder_y.inverse_transform(predicted_medicine_encoded)

    return predicted_medicine[0]


user_symptoms = input("Enter the symptoms you are experiencing (comma separated): ")
user_causes = input("Enter the causes (if known): ")
predicted_medicine = suggest_medicine(user_symptoms, user_causes)
print(f"Suggested Medicine: {predicted_medicine}")
