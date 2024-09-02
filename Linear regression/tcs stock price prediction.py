import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
df = pd.read_csv("TCS.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
plt.scatter(df['Date_ordinal'], df['Close'], color='blue')
plt.xlabel('Date (ordinal)')
plt.ylabel('Close Price')
plt.title('Date vs. Close Price')
plt.show()
sns.displot(df['Close'], kde=True)
plt.title('Distribution of Close Prices')
plt.show()
x = df['Date_ordinal'].values.reshape(-1, 1)
y = df['Close'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
pre = reg.predict(x_test)
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, pre, color='red', label='Predicted')
plt.xlabel('Date (ordinal)')
plt.ylabel('Close Price')
plt.title('Actual vs. Predicted Close Prices')
plt.legend()
plt.show()
d = input("Enter the date to predict (YYYY-MM-DD): ")
d_ordinal = pd.to_datetime(d).toordinal()
prediction = reg.predict(np.array([[d_ordinal]]))
print(f"The predicted Close price for the date {d} is: {prediction[0][0]:.2f}")
