import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Iris Flower Classifier')
st.markdown('''
                Model to classify iris flower into setosa , versicolor,virginica
            ''')
st.header("Plant Features")
col1,col2 = st.columns(2)
with col1:
    st.text('Sepal Characteristics')
    sepal_l=st.slider('Sepal length (cm)',1.0,8.0,0.5)
    sepal_w=st.slider('Sepal width (cm)',2.0,4.4,0.5)
with col2:
    st.text('Petal Characteristics')
    petal_l=st.slider('Petal length (cm)',1.0,8.0,0.5)
    petal_w=st.slider('Petal width (cm)',2.0,4.4,0.5)

# st.button('Predict Type of Iris')

if st.button('Predict Type of Iris'):
    result= predict(np.array([[sepal_l,sepal_w,petal_l,petal_w]]))
    st.markdown('''
                ### The Iris Type is :
                ## " {} "
                '''.format(result[0]))