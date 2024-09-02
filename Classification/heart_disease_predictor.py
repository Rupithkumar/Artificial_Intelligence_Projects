from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load and prepare the data
df = pd.read_csv('heart problem.csv')
df = df.dropna(subset=['TenYearCHD']).reset_index(drop=True)
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot comparison of test vs. predicted values
plt.figure(figsize=(10, 6))
sns.histplot([y_test, y_pred], multiple='dodge', palette='Set2', kde=False)
plt.title('Test vs Predicted Values')
plt.xlabel('Heart Disease Prediction (0 = No, 1 = Yes)')
plt.ylabel('Frequency')
plt.legend(labels=['Test', 'Predicted'])
plt.show()

# Print classification report
print(class_report)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        male = int(request.form['male'])
        age = int(request.form['age'])
        education = float(request.form['education'])
        currentSmoker = int(request.form['currentSmoker'])
        cigsPerDay = float(request.form['cigsPerDay'])
        BPMeds = float(request.form['BPMeds'])
        prevalentStroke = int(request.form['prevalentStroke'])
        prevalentHyp = int(request.form['prevalentHyp'])
        diabetes = int(request.form['diabetes'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = float(request.form['heartRate'])
        glucose = float(request.form['glucose'])

        # Prepare the input data
        user_data = [[male, age, education, currentSmoker, cigsPerDay, BPMeds,
                      prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
                      diaBP, BMI, heartRate, glucose]]

        # Handle missing values using the same imputer
        user_data = imputer.transform(user_data)

        # Normalize the input data
        user_data = scaler.transform(user_data)

        # Predict the outcome using the trained model
        prediction = knn.predict(user_data)

        result = "likely to have heart disease" if prediction[0] == 1 else "unlikely to have heart disease"

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
