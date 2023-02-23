import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the light bulbs dataset
df = pd.read_csv("light_bulbs_dataset.csv")

# Define the features and target
features = df.drop("class", axis=1)
target = df["class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Create the machine learning model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Define the function to save the model
@st.cache
def save_model(model):
    import joblib
    joblib.dump(model, "light_bulbs_model.joblib")

# Define the function to load the model
@st.cache
def load_model():
    import joblib
    return joblib.load("light_bulbs_model.joblib")

# Define the function to make predictions
def predict(model, data):
    predictions = model.predict(data)
    return predictions

# Define the web application
def app():
    st.title("Light Bulb Classifier")

    # Define the sidebar
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select an option", ("Train a new model", "Load a saved model"))

    # Train a new model
    if option == "Train a new model":
        st.header("Train a new model")
        st.write(df)

        # Train the model
        model.fit(features, target)

        # Save the model
        save_model(model)

        # Show the accuracy score
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

    # Load a saved model
    elif option == "Load a saved model":
        st.header("Load a saved model")

        # Load the model
        loaded_model = load_model()

        # Show the accuracy score
        y_pred = loaded_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

    # Make predictions
    st.header("Make a prediction")
    hours = st.number_input("Hours", value=100)
    watts = st.number_input("Watts", value=60)
    prediction = predict(model, [[hours, watts]])
    st.write("Prediction:", prediction[0])
