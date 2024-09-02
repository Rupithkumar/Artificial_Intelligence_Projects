import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart problem.csv')
df = df.dropna(subset=['TenYearCHD']).reset_index(drop=True)
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot([y_test, y_pred], multiple='dodge', palette='Set2', kde=False)
plt.title('Test vs Predicted Values')
plt.xlabel('Heart Disease Prediction (0 = No, 1 = Yes)')
plt.ylabel('Frequency')
plt.legend(labels=['Test', 'Predicted'])
plt.show()
print(class_report)


def predict_heart_disease():
    print("Please enter the following details to predict heart disease:")
    male = int(input("Are you male? (1 = Yes, 0 = No): "))
    age = int(input("Enter your age: "))
    education = float(input("Enter your education level (1-4): "))
    currentSmoker = int(input("Are you a current smoker? (1 = Yes, 0 = No): "))
    cigsPerDay = float(input("Average number of cigarettes per day: "))
    BPMeds = float(input("Are you on blood pressure medication? (1 = Yes, 0 = No): "))
    prevalentStroke = int(input("Have you had a stroke? (1 = Yes, 0 = No): "))
    prevalentHyp = int(input("Do you have hypertension? (1 = Yes, 0 = No): "))
    diabetes = int(input("Do you have diabetes? (1 = Yes, 0 = No): "))
    totChol = float(input("Total cholesterol level (mg/dL): "))
    sysBP = float(input("Systolic blood pressure (mmHg): "))
    diaBP = float(input("Diastolic blood pressure (mmHg): "))
    BMI = float(input("Body Mass Index (BMI): "))
    heartRate = float(input("Heart rate (beats per minute): "))
    glucose = float(input("Glucose level (mg/dL): "))
    user_data = [[male, age, education, currentSmoker, cigsPerDay, BPMeds,
                  prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
                  diaBP, BMI, heartRate, glucose]]
    user_data = imputer.transform(user_data)
    user_data = scaler.transform(user_data)
    prediction = knn.predict(user_data)
    if prediction[0] == 1:
        print("\nPrediction: The person is likely to have heart disease.")
    else:
        print("\nPrediction: The person is unlikely to have heart disease.")


predict_heart_disease()
