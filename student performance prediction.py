import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("study_hours.csv")
print(df)
x=df[["hours_study","attendance"]]
y=df["pass"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
score=accuracy_score(y_test,y_pred)
print("Accuracy score:",score)
example = pd.DataFrame([[7,85]], columns=["hours_study","attendance"])
data=model.predict(example)
if data[0]==1:
    print("student pass")
else:
    print("student fails")
plt.title("student performance prediction")
plt.scatter(df["hours_study"], df["pass"], color="darkgreen")
plt.xlabel("hours studied")
plt.ylabel("pass(0=fails,1=pass)")
plt.show()