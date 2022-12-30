import streamlit as st
import pickle

KNNModel=pickle.load(open('knnclassifier.pkl','rb'))
LRegressionModel=pickle.load(open('Regressionclassifier.pkl','rb'))
NBModel=pickle.load(open('NBclassifier.pkl','rb'))
vectorizer=pickle.load(open('Vectorizer.pickle','rb'))

def Prediction(input_data,option):
    input_data=[input_data]    
    email_feautures=vectorizer.transform(input_data)
    ans=''
    if option=='KNN':
        ans=KNNModel.predict(email_feautures)
    elif option=='Naive-Bayes':
        ans=NBModel.predict(email_feautures)
    elif option=='Logistic Regression':
        ans=LRegressionModel.predict(email_feautures)

    return ans

def main():
    st.set_page_config(page_title="Be Cautious",page_icon=":imp:")
    option = st.selectbox(
    'Algorithm',
    ('Select','Naive-Bayes', 'KNN', 'Logistic Regression'))
    st.title('e-mail Spam Detection web app')
    EnterTheSubject=st.text_input('Subject')
    EnterTheBody=st.text_input('Body')
    EnterTheMail=EnterTheSubject + EnterTheBody
    #code for prediction
    prediction=''
    
    #creating a button for prediction
    if st.button('Email detection'):
        prediction=Prediction(EnterTheMail,option)

    st.success(prediction)

if __name__ == "__main__":
    main()
