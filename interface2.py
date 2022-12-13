import streamlit as st
import pickle

KNNModel=pickle.load(open('knnclassifier.pkl','rb'))
LRegressionmodel=pickle.load(open('Regressionclassifier.pkl','rb'))
vectorizer=pickle.load(open('Vectorizer.pickle','rb'))

def emailPrediction(input_data):
    input_data=[input_data]    
    email_feautures=vectorizer.transform(input_data)
    ans=KNNModel.predict(email_feautures)
    ans2=LRegressionmodel.predict(email_feautures)

    return ans

def main():

    st.set_page_config(page_title="Be Cautious",page_icon=":imp:")
    st.title('e-mail Spam Detection web app')
    EnterTheSubject=st.text_input('Subject')
    EnterTheBody=st.text_input('Body')
    EnterTheMail=EnterTheSubject + EnterTheBody
    #code for prediction
    prediction=''
    
    #creating a button for prediction
    if st.button('Email detection'):
        prediction=emailPrediction(EnterTheMail)
    st.success(prediction)

if __name__ == "__main__":
    main()
