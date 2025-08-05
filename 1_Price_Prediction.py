import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
st.set_page_config(page_title="Viz Demo")
import joblib

# Load the pre-trained model pipeline
with open('pipeline_latest.pkl','rb') as file:
    pipeline = pickle.load(file)

df=pd.read_csv(r'C:\Users\Swapnil\Downloads\gurgaon_properties_post_feature_selection_v2.csv')

df['furnishing_type'] = df['furnishing_type'].replace({0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished'})

    
st.header('Enter your inputs')

# property_type
property_type = st.selectbox('Property Type',['flat','house'])

# sector
sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))

bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))

built_up_area = float(st.number_input('Built Up Area'))

servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))
store_room = float(st.selectbox('Store Room',[0.0, 1.0]))

furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
# furnishing_type = str(furnishing_type)  # Ensure it's a string


luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

if st.button('Predict'):

    # form a dataframe
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)
    #one_df['bedRoom'] = pd.to_numeric(one_df['bedRoom'], errors='coerce')
    #print(one_df.isnull().sum())


    numeric_cols = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
    one_df[numeric_cols] = one_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Ensure categorical columns are correct type
    categorical_cols = ['property_type', 'sector', 
                        'balcony', 'agePossession', 'luxury_category', 'floor_category','furnishing_type']
    one_df[categorical_cols] = one_df[categorical_cols].astype('object')

    output=pipeline.predict(one_df)
    # predict
    base_price = np.expm1(output)[0]
    low = base_price - 0.22
    high = base_price + 0.22

    # display
    st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))

    
