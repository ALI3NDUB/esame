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
 
       
    st.title('Company linear')
    st.header('This app allows you to explore the Iris dataset and visualize the data using various plots.')
    st.subheader("DataFrame")
    
    path="https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/Company.csv"
    df = pd.read_csv(path)

    st.dataframe(df)

    X = df.drop(columns="Sales")
    y = df["Sales"]

    st.subheader('Input')
    rd_spend = st.slider('TV', min_value=0, max_value=500, value=150)
    admin = st.slider('Radio', min_value=0, max_value=500, value=150)
    marketing = st.slider('Newspaper', min_value=0, max_value=500, value=150)

    input_data = {'TV': rd_spend,
                  'Radio': admin,
                  'Newspaper': marketing}

    input_df = pd.DataFrame(input_data, index=[0])
    prediction = new_model.predict([[rd_spend,admin,marketing]])

    st.subheader('Output')
    st.write(f'Audience previsto: {prediction[0]}')

    # Plotly graph
    fig = px.scatter_3d(df, x='TV', y='Radio', z='Newspaper', color_discrete_sequence=['red'])
    fig.update_layout(
        title="Regresione multipla",
        scene=dict(
            xaxis_title='TV',
            yaxis_title='Radio',
            zaxis_title='Newspaper'
        )
    )
    #fig.add_trace(px.scatter_3d(input_df, x='TV', y='Radio', z='Newspaper', color='Radio', opacity='[1]').data[0])
    #st.plotly_chart(fig, use_container_width=True)
    
    #fig.add_trace(px.scatter_3d(input_df, x='TV', y='Radio', z='Newspaper', color_discrete_sequence=['Yellow'], size='Newspaper').data[0])  
    #st.plotly_chart(fig, use_container_width=True)
    
    fig.add_trace(px.scatter_3d(input_df, x='TV', y='Radio', z='Newspaper', color='Newspaper', color_continuous_scale='oryel', opacity=0.5).data[0])
    fig.update_layout(scene=dict(bgcolor='rgb(145, 160, 155)'))
    st.plotly_chart(fig, use_container_width=True)

    predictions = new_model.predict(X)
    custom_predictions = new_model.predict([[rd_spend, admin, marketing]])[0]
    st.write(custom_predictions)

    if st.button('Scarica Excel'):
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        input_df.to_excel(writer, index=False, sheet_name='Input')
        pd.DataFrame(prediction, columns=['Profitto previsto']).to_excel(writer, index=False, sheet_name='Output')
        writer.save()
        output.seek(0)
        st.download_button(label="Download", data=output, file_name='output.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',)

if __name__ == '__main__':
    main()
    
# streamlit run app.py        