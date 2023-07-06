from flask import Flask,render_template,request
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB

app= Flask(__name__)

@app.route("/")
def home():
    df1=pd.read_csv('./titanic/train.csv')
    df1.dropna(inplace=True)
    df1.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)

    inputs = df1.drop('Survived',axis='columns')
    target = df1.Survived
    dummies = pd.get_dummies(inputs.Sex)
    inputs = pd.concat([inputs,dummies],axis='columns')
    inputs.drop(['Sex'],axis='columns',inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)
    model = GaussianNB()
    model.fit(X_train,y_train)
    pickle.dump(model,open("iris.pkl","wb"))

    return render_template("home.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    pclass = request.form['pclass']
    Age = request.form['Age']
    Fare = request.form['Fare']
    Sex = request.form['Sex']
    female=0
    male=0
    if(Sex=='female'):
        female=1
    else : male=1
    form_array = np.array([[pclass,Age,Fare,female,male]])
    model = pickle.load(open("iris.pkl","rb"))
    prediction = model.predict(form_array)[0]
    if prediction ==0:
        result="Sorry you wouldn't have survived it"
    else:
        result="you would have survived it"
    return render_template("result.html",result = result)

if __name__ == "__main__":
    app.run(debug=True)
