import pandas as pd, pickle, streamlit as st, numpy as np
import streamlit.components.v1 as components

st.set_page_config('Customer Intent Predictor',layout='wide')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.title('Customer Indent Prediction')

c1,_,c2 = st.columns([0.5,0.1,0.4])

df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('pipe.pkl','rb'))

with c1:
  st.markdown('### Predictors', unsafe_allow_html=True)
  with st.container(border=True):
    product_cat = st.selectbox(label = 'Select the product category',
                               options = list(df['ProductCategory'].unique()))
    
    product_brand = st.selectbox(label = 'Select the product brand',
                               options = list(df['ProductBrand'].unique()))
    
    price = st.number_input(label='Input the price', value=df['ProductPrice'].quantile(q=0.5))
    
    age = st.number_input(label = 'Enter the age', min_value=df['CustomerAge'].min())
    
    gender = st.selectbox(options=['Male','Female'], label='Select the gender')
    
    if gender == 'Male':
      gender = 1
    else:
      gender = 0
      
    freq = st.number_input(label = 'Enter the frequency', min_value=df['PurchaseFrequency'].min())
    
    satis = st.number_input(label = 'Enter the Satisfaction', min_value=df['CustomerSatisfaction'].min())
    
    q = np.array([[product_cat,product_brand, price, age, gender, freq, satis]])
    pred_data = pd.DataFrame(q, columns=['ProductCategory', 'ProductBrand', 'ProductPrice', 'CustomerAge',
       'CustomerGender', 'PurchaseFrequency', 'CustomerSatisfaction'])
    
  pred = st.button(label = 'Predict')
    
  if pred:
    prediction_result = model.predict(pred_data)
      
    if prediction_result==1:
      st.success('Buying Intent')
    else:
      st.error('Not Buying Intent')
        
with c2:

  
  st.markdown('#### Customer Intent')
  st.write('Customer intent refers to the or goal that drives a customer’s actions or interactions with a business. It’s essential for businesses to understand customer intent to provide personalized and relevant experiences. ')
  
  st.markdown('#### Objective')
  st.write(f'Predicting the customer intent of buying the product')
  
  st.markdown('#### Why Important')
  st.write('Predicting customer intent is incredibly valuable for businesses and organizations. Understanding customer intent allows businesses to tailor their interactions, content, and recommendations, resulting in personalized experiences that enhance customer satisfaction and loyalty.')
  
  st.markdown('#### Model')
  st.write('Logistic Regression is a statistical model used for binary classification. It estimates the probability that an event (e.g., a customer making a purchase) occurs based on a set of independent variables . Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities between 0 and 1. The model uses the sigmoid function to map real-valued inputs into this probability range. The goal is to predict whether a customer will take a specific action.')
  
