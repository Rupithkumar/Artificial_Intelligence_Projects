import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
df=pd.read_csv("Housing.csv")
plt.scatter(df.area,df.price,color="blue")
plt.show()
sns.displot(df.price,kde=True)
plt.show()
reg=linear_model.LinearRegression()
x=df['area'].values.reshape(-1,1)
y=df['price'].values.reshape(-1,1)
reg.fit(x,y)
plt.scatter(x,y,color='gray')
plt.plot(df.area,reg.predict(df[['area']]),color='red',linewidth=3)
plt.show()
p=int(input("Enter the area to predict the price: "))
prediction=reg.predict(np.array([[p]]))
print(f"The price of the area {p} predicted is: {prediction[0][0]}")