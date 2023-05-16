import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.express as px
import mlem
import matplotlib.pyplot as plt
import io
import mlem

#Import warnings for sklearn
import warnings
warnings.filterwarnings('ignore')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.londonart.it/public/images/2020/restanti-varianti/20043-01.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main(): 
    new_model = mlem.api.load('model.mlem')
      
    add_bg_from_url()
 
    st.title('Immobili')
    st.header('Questa web-app sfrutta Streamlit per utilizzare un modello di machine learning')
    st.subheader("DataFrame Originale")
    
    path="https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/immobili.csv"
    df = pd.read_csv(path)

    st.dataframe(df)

    y = df["medv"]
    X = df.drop(columns="medv")

    st.subheader('Input')
    sl1 = st.slider('indus', min_value=0, max_value=500, value=150)
    sl2 = st.slider('age', min_value=0, max_value=500, value=150)
    sl3 = st.slider('tax', min_value=0, max_value=500, value=150)

    input_data = {'indus': sl1,
                  'age': sl2,
                  'tax': sl3}

    input_df = pd.DataFrame(input_data, index=[0])
    prediction = new_model.predict([[sl1,sl2,sl3]])

    st.subheader('Output')
    st.write(f'Predizione previsto: {prediction[0]}')

    # Plotly graph
    fig = px.scatter_3d(df, x='lstat', y='rm', z='ptratio', color_discrete_sequence=['red'])
    fig.update_layout(
        title="Plot",
        scene=dict(
            xaxis_title='lstat',
            yaxis_title='rm',
            zaxis_title='ptradio'
        )
    )  

    custom_predictions = new_model.predict([[sl1, sl2, sl3]])[0]
    st.write(custom_predictions)

if __name__ == '__main__':
    main()
    
# streamlit run app.py        