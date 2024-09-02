import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv("weather.csv")
plt.xlabel("Minimum temperature")
plt.ylabel("Maximum temperature")
plt.scatter(df.MinTemp, df.MaxTemp, color="red")
plt.show()
sns.displot(df['MaxTemp'], kde=True)
plt.show()
x = df['MinTemp'].values.reshape(-1, 1)
y = df['MaxTemp'].values.reshape(-1, 1)
reg = linear_model.LinearRegression()
reg.fit(x,y)
plt.scatter(df.MinTemp,df.MaxTemp,color="gray")
plt.plot(df.MinTemp,reg.predict(df[['MinTemp']]),color="red",linewidth=5)
plt.show()
i = int(input("Enter the minimum temperature (in celsius): "))
prediction = reg.predict(np.array([[i]]))
print(f"Predicted MaxTemp for MinTemp={i} in celsius : {prediction[0][0]} C")