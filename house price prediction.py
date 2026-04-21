import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
df=pd.read_csv("house_data.csv")
print(df)
x=df[["area"]]
y=df[["price"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
score=r2_score(y_test,y_pred)
print("r2 score:",score)
plt.title("House price prediction")
plt.scatter(x,y,color="red")
plt.plot(x,y,color="blue")
plt.xlabel("area")
plt.ylabel("price")
plt.grid(True)
plt.show()