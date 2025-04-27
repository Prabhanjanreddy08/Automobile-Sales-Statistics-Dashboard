# Automobile-Sales-Statistics-Dashboard

This **Automobile Sales Statistics Dashboard** is an interactive data visualization tool built with **Streamlit**. It provides users with insights into **automobile sales data**. The dashboard allows exploration of historical sales trends, the impact of recession periods on sales, and the ability to make predictions for future trends using machine learning models.

The dashboard is designed to aid data analysis by exploring how various factors like recession periods, unemployment rates, and advertising expenditures influence automobile sales. It also provides predictive models to forecast sales, advertisement expenditures, and vehicle types based on historical data.

## Features

### 1. **Yearly Statistics**
This feature allows users to explore automobile sales data for a specific year. The following insights are available:
- **Yearly Average Sales**: Visualize the average automobile sales by year.
- **Monthly Sales Patterns**: See how automobile sales fluctuate on a monthly basis.
- **Vehicle Type Performance**: View average sales by vehicle type for the selected year, allowing comparisons across categories.
- **Advertisement Expenditure by Vehicle Type**: Explore how advertisement expenditure varies by vehicle type in the selected year.

### 2. **Recession Period Statistics**
This feature allows users to analyze how automobile sales have been affected during recession periods. Key visualizations include:
- **Sales Fluctuation During Recession**: Explore how automobile sales have changed during recession years.
- **Vehicle Type Comparison**: See how different vehicle types have performed during recession periods.
- **Advertising Expenditure Share**: Understand how advertising expenditure is distributed across vehicle types during recessions.
- **Unemployment Rate Effects**: Visualize how unemployment rates have impacted automobile sales and vehicle types during recession years.

### 3. **Prediction Models**
The dashboard includes three predictive models to forecast future trends:
- **Model 1: Predict Automobile Sales**: A **linear regression model** that predicts future automobile sales based on historical data. Users can forecast sales for future years (e.g., 2024, 2025, 2026).
- **Model 2: Predict Advertisement Expenditure**: A **linear regression model** that estimates advertisement expenditure based on automobile sales figures. Users can input sales data to predict the corresponding ad spend.
- **Model 3: Predict Vehicle Type**: A **random forest classifier** that predicts the vehicle type based on sales figures and advertisement expenditures. This model predicts the type of vehicle likely to perform well based on given sales and advertising data.

## Technologies Used

The dashboard leverages several powerful Python libraries for data manipulation, visualization, and machine learning:

- **Streamlit**: For creating the interactive web dashboard.
- **Pandas**: For data manipulation and analysis.
- **Plotly**: For interactive visualizations (line charts, bar charts, pie charts).
- **Scikit-learn**: For machine learning models (Linear Regression, Random Forest Classifier).
- **Seaborn** and **Matplotlib**: For additional data visualizations.


## How to Use
Yearly Statistics: In the sidebar, select "Yearly Statistics" and choose a specific year from the dropdown list. This will display various graphs showing sales trends, vehicle performance, and advertisement spending for the selected year.

Recession Period Statistics: Select "Recession Period Statistics" from the sidebar. This will show you how sales and other factors such as advertising expenditure and unemployment rates have been impacted during recession periods.

Prediction Models: Select "Prediction Models" from the sidebar. This section includes three tabs:

- **Model 1: Predict Sales: Enter future years to predict automobile sales based on historical data.

- **Model 2: Predict Ad Spend: Enter automobile sales figures to predict the corresponding advertisement expenditure.

- **Model 3: Predict Vehicle Type: Enter automobile sales and ad expenditure to predict the type of vehicle that is likely to perform well.


## Acknowledgements
IBM Developer Skills Network for the original dataset used in this project.
Streamlit, Pandas, Plotly, Scikit-learn, Seaborn, and Matplotlib for providing the necessary tools and libraries for this project.
