import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

#load data
data=pd.read_csv('creditcard.csv')

#seprate legitimate and fraudulent transction
legit=data[data.Class==0]
fraud=data[data.Class==1]

#undersample legitimate transction to balance the classes
legit_sample=legit.sample(n=len(fraud),random_state=2)
data=pd.concat([legit_sample,fraud],axis=0)

# split data into training and testing sets
x=data.drop(columns="Class",axis=1)
y=data["Class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

#train logistic regression model
model=LogisticRegression()
model.fit(x_train,y_train)

#evaluate model performance
train_acc=accuracy_score(y_train, model.predict(x_train))
test_acc=accuracy_score(y_test, model.predict(x_test))

#web app
st.title("Credit Card Fraud detection model")
input_df=st.text_input('Enter All Parameter Values')
input_df_splited=input_df.split(',')

submit=st.button("Submit")

if submit:
    features=np.asarray(input_df_splited,dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))

    if prediction[0] == 0:
        st.write("This transction is not fraud")
    else:
        st.write("This is fraud transction")
