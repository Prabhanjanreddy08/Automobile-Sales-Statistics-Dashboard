import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv')

# Title
st.title("Automobile Sales Statistics Dashboard")

# Sidebar for selection
st.sidebar.header("Select Statistics")
statistics_option = st.sidebar.selectbox(
    "Choose a Report Type:",
    ['Yearly Statistics', 'Recession Period Statistics', 'Prediction Models']
)

# List of years
year_list = [i for i in range(1980, 2024)]

# Year selection (only if Yearly Statistics selected)
if statistics_option == 'Yearly Statistics':
    selected_year = st.sidebar.selectbox("Select Year", year_list)
else:
    selected_year = None

st.write(f"### {statistics_option}")

# Recession Statistics
if statistics_option == 'Recession Period Statistics':
    recession_data = data[data['Recession'] == 1]

    # Chart 1 - Average Sales over Recession Period
    yearly_rec = recession_data.groupby('Year')['Automobile_Sales'].mean().reset_index()
    fig1 = px.line(yearly_rec, x='Year', y='Automobile_Sales',
                   title="Average Automobile Sales Fluctuation Over Recession Period")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 - Average Vehicles Sold by Type
    average_sales = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
    fig2 = px.bar(average_sales, x='Vehicle_Type', y='Automobile_Sales',
                  title="Average Number of Vehicles Sold by Vehicle Type")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 - Expenditure Share by Vehicle Type
    exp_rec = recession_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
    fig3 = px.pie(exp_rec, names='Vehicle_Type', values='Advertising_Expenditure',
                  title="Expenditure Share by Vehicle Type During Recession")
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4 - Unemployment Rate effect
    unemp_data = recession_data.groupby(['unemployment_rate', 'Vehicle_Type'])['Automobile_Sales'].mean().reset_index()
    fig4 = px.bar(unemp_data, x='unemployment_rate', y='Automobile_Sales', color='Vehicle_Type',
                  labels={'unemployment_rate': 'Unemployment Rate', 'Automobile_Sales': 'Average Automobile Sales'},
                  title='Effect of Unemployment Rate on Vehicle Type and Sales')
    st.plotly_chart(fig4, use_container_width=True)

# Yearly Statistics
elif statistics_option == 'Yearly Statistics' and selected_year:
    yearly_data = data[data['Year'] == selected_year]

    # Chart 1 - Yearly Average Automobile Sales
    yas = data.groupby('Year')['Automobile_Sales'].mean().reset_index()
    fig1 = px.line(yas, x='Year', y='Automobile_Sales',
                   title='Yearly Average Automobile Sales')
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 - Monthly Sales Average
    mas = data.groupby('Month')['Automobile_Sales'].mean().reset_index()
    fig2 = px.line(mas, x='Month', y='Automobile_Sales',
                   title='Average Monthly Automobile Sales')
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 - Vehicle Type Sales in Selected Year
    avr_vdata = yearly_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
    fig3 = px.bar(avr_vdata, x='Vehicle_Type', y='Automobile_Sales',
                  title=f'Average Vehicles Sold by Vehicle Type in {selected_year}')
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4 - Advertisement Expenditure
    exp_data = yearly_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
    fig4 = px.pie(exp_data, names='Vehicle_Type', values='Advertising_Expenditure',
                  title=f'Total Advertisement Expenditure by Vehicle Type in {selected_year}')
    st.plotly_chart(fig4, use_container_width=True)

# Prediction Models
elif statistics_option == 'Prediction Models':

    tab1, tab2, tab3 = st.tabs(["Model 1: Predict Sales", "Model 2: Predict Ad Spend", "Model 3: Predict Vehicle Type"])

    with tab1:
        st.write("#### Model 1: Predict Automobile Sales based on Year")
        model1_data = data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        X = model1_data['Year'].values.reshape(-1, 1)
        y = model1_data['Automobile_Sales'].values

        model1 = LinearRegression()
        model1.fit(X, y)

        future_years = np.array([2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)
        future_sales = model1.predict(future_years)

        future_df = pd.DataFrame({
            'Year': future_years.flatten(),
            'Predicted_Sales': future_sales
        })

        fig5 = px.line(model1_data, x='Year', y='Automobile_Sales', title='Historical and Predicted Automobile Sales')
        fig5.add_scatter(x=future_df['Year'], y=future_df['Predicted_Sales'], mode='lines+markers', name='Predicted Sales')
        st.plotly_chart(fig5, use_container_width=True)

        st.dataframe(future_df)

    with tab2:
        st.write("#### Model 2: Predict Advertisement Expenditure based on Sales")

        # Prepare data
        ad_data = data[['Automobile_Sales', 'Advertising_Expenditure']].dropna()

        X2 = ad_data[['Automobile_Sales']]
        y2 = ad_data['Advertising_Expenditure']

        # Model (Option 1: Linear Regression + Clip negatives)
        model2 = LinearRegression()
        model2.fit(X2, y2)

        # User input
        sales_input = st.number_input("Enter Automobile Sales for Advertisement Prediction:", min_value=0)

        if sales_input > 0:
            # Predict
            predicted_ad_exp = model2.predict(np.array([[sales_input]]))
            predicted_ad_exp = np.maximum(predicted_ad_exp, 0)  # Avoid negative predictions

            st.success(f"Predicted Advertisement Expenditure: ${predicted_ad_exp[0]:,.2f}")

        # Visualization: Sales vs Advertising Expenditure
        st.write("#### Relationship between Sales and Advertisement Expenditure")
        fig6 = px.scatter(
            ad_data,
            x='Automobile_Sales',
            y='Advertising_Expenditure',
            trendline='ols',
            title="Sales vs Advertisement Expenditure",
            labels={'Automobile_Sales': 'Automobile Sales', 'Advertising_Expenditure': 'Ad Expenditure'}
        )
        st.plotly_chart(fig6, use_container_width=True)


    with tab3:
        st.write("#### Model 3: Predict Vehicle Type based on Sales and Advertising Expenditure")

        vt_data = data[['Automobile_Sales', 'Advertising_Expenditure', 'Vehicle_Type']].dropna()

        # Encode Vehicle_Type
        le = LabelEncoder()
        vt_data['Vehicle_Type_Encoded'] = le.fit_transform(vt_data['Vehicle_Type'])

        X3 = vt_data[['Automobile_Sales', 'Advertising_Expenditure']]
        y3 = vt_data['Vehicle_Type_Encoded']

        model3 = RandomForestClassifier()
        model3.fit(X3, y3)

        sales_value = st.number_input("Enter Sales for Prediction (Vehicle Type Model):", min_value=0)
        ad_exp_value = st.number_input("Enter Ad Expenditure for Prediction (Vehicle Type Model):", min_value=0)

        if sales_value > 0 and ad_exp_value > 0:
            pred_vehicle_encoded = model3.predict(np.array([[sales_value, ad_exp_value]]))
            pred_vehicle = le.inverse_transform(pred_vehicle_encoded)
            st.success(f"Predicted Vehicle Type: {pred_vehicle[0]}")

