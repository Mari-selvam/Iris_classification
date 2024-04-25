import streamlit as st
import pandas as pd
from knn import prediction
import time


st.title('Iris Classification 💐 🌺')
data = pd.read_csv('iris.csv')

st.write('Iris classification using K-Nearest Neighbors (KNN) is a method of categorizing iris flowers into different species based on their botanical features. KNN operates on the principle of proximity, where it classifies a data point (in this case, an iris flower) by examining the class labels of its nearest neighbors in the feature space.')
st.write('---')

st.subheader('Iris data')
st.dataframe(data)
st.write('---')

st.subheader('Iris data')





st.scatter_chart(data ,x='sepal_length' ,y='sepal_width',color='species' ,height=500 ,width=500)

st.divider()
st.title('Demo : ')


# with st.spinner('Wait for it...'):
#     time.sleep(4)
    
# st.success('Done!')

sepal_length = st.number_input('sepal length :')
sepal_width = st.number_input('sepal width :')
petal_length = st.number_input('petal length :')
petal_width = st.number_input('petal width :')





if st.button('Predict'):
    st.toast('Analysis')
    time.sleep(.5)
    output = prediction([sepal_length,sepal_width,petal_length,petal_width])
    
    st.toast('Predicting!')
    time.sleep(.5)
    
    label = {1:'setosa', 2:'versicolor', 3:'virginica'}
    st.subheader(label[int(output)])
    
    
    if label[int(output)] == 'virginica':
        st.image('Iris_virginica_2.jpg')
        st.write('The Predicted output was virginica')

    elif label[int(output)] == 'setosa':

        st.image('Irissetosa1.jpg')
        st.write('The Predicted output was  setosa')

        
    elif label[int(output)] == 'versicolor':
        st.image('versicolor.jpeg')
        st.write('The Predicted output was versicolor')
        

        

    st.toast('Output!', icon='🎉')

    
    
    
    



