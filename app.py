import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = {

    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon',
                  'Evening', 'Morning'],
    'SleepQuality': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']

}
df = pd.DataFrame(data)
#Encode
df_encoded = df.copy()
label_encoder = {}
for column in df.columns:
    if df [column].dtype == "object":
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df[column])
        label_encoder[column] = le

#Feature and target
X = df_encoded.drop("BuyCoffee", axis = 1)
y = df_encoded["BuyCoffee"]

#train decision tree

model = DecisionTreeClassifier(criterion = "entropy")
model.fit(X,y)
#Visualization with Matplotlib
plt.figure(figsize = (12,6))
plot_tree(model, feature_names = X.columns, class_names = label_encoder["BuyCoffee"].classes_, filled= True)
plt.show()
st.title("BuyCoffee Prediction with ID3 Decision Tree")


def user_input():
    weather = st.sidebar.selectbox("Weather", df['Weather'].unique())
    TimeOfDay = st.sidebar.selectbox("TimeOfDay", df['TimeOfDay'].unique())
    SleepQuality = st.sidebar.selectbox("SleepQuality", df['SleepQuality'].unique())
    Mood = st.sidebar.selectbox("Mood", df['Mood'].unique())

    return pd.DataFrame([[weather, TimeOfDay, SleepQuality, Mood]],
                        columns=['Weather', 'TimeOfDay', 'SleepQuality', 'Mood'])


input_df = user_input()

#Encode input
input_encoded = input_df.copy()
for col in input_encoded.columns:
    input_encoded[col] = label_encoder[col].transform(input_encoded[col])
# Prediction
prediction = model.predict(input_encoded)[0]
prediction_label = label_encoder['BuyCoffee'].inverse_transform([prediction])[0]
st.subheader("Prediction:")
st.success(f"The model predicts: {prediction_label}")
st.subheader("Input Values:")
st.write(input_df)
st.subheader("Training Data:")
st.dataframe(df)