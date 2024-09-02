import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
df=pd.read_csv('score_updated.csv')
plt.scatter(df.Hours,df.Scores,color="red")
plt.title("Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
sns.displot(df['Hours'],kde=True)
plt.show()
x=df['Hours'].values.reshape(-1,1)
y=df['Scores'].values.reshape(-1,1)
reg=linear_model.LinearRegression()
reg.fit(x,y)
plt.scatter(x,y,color='gray')
plt.plot(df.Hours,reg.predict(df[['Hours']]))
plt.show()
h=int(input('Enter you the hours that you studied: '))
prediction=reg.predict(np.array([[h]]))
print(f'The score that can be scored for {h} hours of study is: {prediction[0][0]}')
