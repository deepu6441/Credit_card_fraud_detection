import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_linear import LogisticRegression
from sklearn.matrics import accuracy_score
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
train_acc=accuracy_score(model.predict(x_train,y_train))
test_acc=accuracy_score(model.predict(x_test,y_test))

#create Streamlit app
st.title("Credit card Fraud detection Model")
st.write("Enter the following feature to check if the transction is legitimate or fraudulent")

#create input fields for user to enter feature valves
input_df=st.text_input('Input all features')
input_df_lst=input_df.split(',')

#create a button to submit input and getprediction
submit=st.button("Sunmit")

if submit:
    features=np.array(input_df_lst,dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))

    #display result
    if prediction[0]==0:
        st.write("This transction is not fraud")
    else:
        st.write("This is fraud transction")