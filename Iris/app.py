import streamlit as st
import pickle
import numpy as np


st.title('Iris Species Prediction')

with open('iris_model.pkl','rb') as f:
    model = pickle.load(f)

speal_leaf_length = st.slider('Sepal Length (cm)', min_value=1.0, max_value=8.0)
speal_leaf_width = st.slider('Sepal Width (cm)', min_value=1.0, max_value=8.0)
petal_length = st.slider('Petal Length (cm)', min_value=1.0, max_value=8.0)
petal_width = st.slider('Petal Width (cm)', min_value=0.0, max_value=4.0)



if st.button('Predict'):
    input_data = np.array([[speal_leaf_length, speal_leaf_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    st.write(f'The predicted species is: {species[prediction[0]]}')