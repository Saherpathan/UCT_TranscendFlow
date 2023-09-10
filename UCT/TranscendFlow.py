import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the training dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Load the test dataset
test_data = pd.read_csv("test_BdBKkAj.csv")

# Convert DateTime column to datetime format
train_data['DateTime'] = pd.to_datetime(train_data['DateTime'])
test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])

# Function to adjust predictions based on special occasions, weekends, and holidays
def adjust_predictions(predictions, datetime_series):
    adjusted_predictions = predictions.copy()
    for i, datetime_value in enumerate(datetime_series):
        if datetime_value.dayofweek >= 5:  # 5 and 6 represent Saturday and Sunday
            # Adjust the prediction here as needed
            adjusted_predictions[i] *= 1.2  # Adjust by a factor of 1.2 for weekends
            # You can add more conditions for special occasions and holidays here
    return adjusted_predictions

# Calculating average vehicles for training data, for data grouping
avg_traffic = train_data.groupby(['Junction', pd.Grouper(key='DateTime', freq='D')])['Vehicles'].mean().reset_index()

# Set page title and description
st.set_page_config(page_title="TranscendFlow", page_icon="✅")
st.title("TranscendFlow: Smart City Traffic Patterns")

# Sidebar for user inputs
st.sidebar.title("Options")
junction_option = st.sidebar.selectbox("Select Junction", avg_traffic['Junction'].unique())

# Prediction using Linear Regression for the selected junction
junction_data = avg_traffic[avg_traffic['Junction'] == junction_option]

# Subtitle for actual data
st.subheader(f'Junction {junction_option} - Actual Traffic Data')

# Plot actual data (for training data)
st.line_chart(junction_data.set_index('DateTime')['Vehicles'], use_container_width=True)

# Prediction using Linear Regression
X = pd.to_numeric(junction_data['DateTime']).values.reshape(-1, 1)
y = junction_data['Vehicles'].values
model = LinearRegression()
model.fit(X, y)

# Generate predictions for test data
test_X = pd.to_numeric(test_data[test_data['Junction'] == junction_option]['DateTime']).values.reshape(-1, 1)
test_predictions = model.predict(test_X)

# Adjust predictions based on special occasions, weekends, and holidays
test_predictions_adjusted = adjust_predictions(test_predictions,
                                               test_data[test_data['Junction'] == junction_option]['DateTime'])

# Subtitle for predicted data
st.subheader(f'Junction {junction_option} - Predicted Traffic Data')

# Plot predicted data (for test data)
st.line_chart(
    pd.Series(test_predictions_adjusted, index=test_data[test_data['Junction'] == junction_option]['DateTime']),
    use_container_width=True)

# Subtitle for adjusted test predictions
st.subheader(f'Junction {junction_option} - Adjusted Test Predictions:')

# Display the adjusted test predictions
st.write(test_predictions_adjusted)
