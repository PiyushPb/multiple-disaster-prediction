import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import folium_static

def home_section():
    st.title("Home Section")
    st.write("Welcome to the Home Section!")

def kerala_flood_section():
    st.title("Kerala Flood Section")
    st.write("This section provides information about the Kerala flood.")

    # Load models
    with open('./models/flood_model.pkl', 'rb') as file:
        loaded_models = pickle.load(file)

    # Input field for the user to provide data as a single string
    st.subheader("Enter Data for Prediction (comma-separated):")
    data_input = st.text_input("Enter values separated by commas (e.g., 2022,1.2,3.4,5.6,...)", "2023, 28.7, 44.7, 51.6, 160, 174.7, 824.6, 743, 357.5, 197.7, 266.9, 350.8, 48.4")

    # Display the input data in tabular format
    st.subheader("Input Data:")
    input_data = {
        'Feature': ['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'],
        'Value': [float(value.strip()) for value in data_input.split(',')]
    }
    input_data_df = pd.DataFrame(input_data)
    st.table(input_data_df)

    # Button to trigger predictions
    if st.button("Generate Predictions"):
        # Process the input string into a list of float values
        data_point = [float(value.strip()) for value in data_input.split(',')]

        new_data_point = np.array(data_point).reshape(1, -1)

        # Display predictions and accuracy in a tabular format
        st.subheader("Model Predictions and Accuracy:")
        st.write("Here are the predictions and accuracy from different models:")

        predictions_table = {
            'Model': [],
            'Prediction': [],
            'Accuracy': []
        }


        data = pd.read_csv('../data/kerala.csv')

        # Data preprocessing (replace this with your own preprocessing steps)
        data['FLOODS'].replace(['YES', 'NO'], [1, 0], inplace=True)
        x = data.iloc[:, 1:14]
        y = data.iloc[:, -1]

        # Train-test split with a fixed random seed
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


        models = []
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('LogisticRegression', LogisticRegression()))
        models.append(('SVC', SVC()))
        models.append(('DecisionTree', DecisionTreeClassifier()))
        models.append(('RandomForest', RandomForestClassifier()))

        names = []
        scores = []
        for name, model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores.append(accuracy_score(y_test, y_pred))
            names.append(name)
        tr_split = pd.DataFrame({'Name': names, 'Score': scores})
        tr_split = tr_split.sort_values(by='Score', ascending=False)


        for model_name, model in loaded_models.items():
            predictions = model.predict(new_data_point)

            matching_rows = tr_split[tr_split['Name'] == model_name]

            if not matching_rows.empty:
                accuracy_value = matching_rows['Score'].values[0]
            else:
                # Handle the case when there is no matching model name
                accuracy_value = None

            predictions_table['Model'].append(model_name)
            predictions_table['Prediction'].append('YES' if predictions[0] == 1 else 'NO')
            predictions_table['Accuracy'].append(accuracy_value)

        predictions_df = pd.DataFrame(predictions_table)
        st.table(predictions_df)


def earthquake_section():
    st.title("Earthquake Section")
    st.write("This section provides information about the Earthquake.")

    # Load the trained models
    with open('./models/earthquake_model.pkl', 'rb') as file:
        loaded_models = pickle.load(file)

    # Input field for the user to provide data as a single string
    st.subheader("Enter Data for Prediction (comma-separated):")
    data_input = st.text_input("Enter values separated by commas (e.g., 2022,1.2,3.4,5.6,...)", "29.06,77.42,5")

    st.subheader("Input Data:")
    input_data = {
        'Feature': ['Latitude', 'Longitude', 'Depth'],
        'Value': [float(value.strip()) for value in data_input.split(',')]
    }
    input_data_df = pd.DataFrame(input_data)
    st.table(input_data_df)

    # Button to display the map
    if st.button("Show Map"):
        # Display the map
        st.subheader("Earthquake Location on Map:")
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Centered around India

        # Add a marker for the provided latitude and longitude
        latitude, longitude = input_data['Value'][:2]  # Extract the first two values as latitude and longitude
        folium.Marker(location=[latitude, longitude],
                      popup="Earthquake Location",
                      icon=folium.Icon(color='red')).add_to(m)

        # Display the map in the Streamlit app
        folium_static(m)

# duhdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd


        data_point = [float(value.strip()) for value in data_input.split(',')]

        new_data_point = np.array(data_point).reshape(1, -1)

        st.subheader("Model Predictions and Accuracy:")
        st.write("Here are the predictions and accuracy from different models:")

        predictions_table = {
            'Model': [],
            'Prediction': [],
            'Accuracy': []
        }

        data = pd.read_csv("../data/earthquake.csv")

        data = np.array(data)

        X = data[:, 0:-1]
        y = data[:, -1]
        y = y.astype('int')
        X = X.astype('int')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        from sklearn.metrics import accuracy_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier

        ert_models = []
        ert_models.append(('KNN', KNeighborsClassifier()))
        ert_models.append(('LR', LogisticRegression()))
        ert_models.append(('SVC', SVC()))
        ert_models.append(('DT', DecisionTreeClassifier()))
        ert_models.append(('RF', RandomForestClassifier()))

        names = []
        scores = []
        for name, model in ert_models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
            names.append(name)

        # Create a DataFrame to display the results
        tr_split = pd.DataFrame({'Name': names, 'Score': scores})
        tr_split = tr_split.sort_values(by='Score', ascending=False)


        for inx, (model_name, model) in enumerate(loaded_models.items()):
            predictions = model.predict(new_data_point)

            matching_rows = tr_split[tr_split['Name'] == model_name]

            accuracy_value = matching_rows['Score'].values[0] if not matching_rows.empty else None

            predictions_table['Model'].append(model_name)
            predictions_table['Prediction'].append(predictions[0])
            predictions_table['Accuracy'].append(scores[inx])

        predictions_df = pd.DataFrame(predictions_table)
        st.table(predictions_df)


def rainfall_section():
    st.title("Rainfall prediction")
    st.write("This section provides information about the rainfall.")

    with open('./models/rainfall_models.pkl', 'rb') as file:
        loaded_models = pickle.load(file)

    st.subheader("Enter Data for Prediction (comma-separated):")
    data_input = st.text_input("Enter values separated by commas (e.g., 2020,18,16,65,1013,6,8)", "2020,18,16,65,1013,6,8")

    st.subheader("Input Data:")
    input_data = {
        'Feature': ['YEAR', 'TEMPAVG', 'DPavg', 'Humidity avg', 'SLPavg', 'visibilityavg', 'windavg'],
        'Value': [float(value.strip()) for value in data_input.split(',')]
    }

    input_data_df = pd.DataFrame(input_data)
    st.table(input_data_df)

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error

    if st.button("Generate Predictions"):
        data_point = [float(value.strip()) for value in data_input.split(',')]

        new_data_point = np.array(data_point).reshape(1, -1)

        st.subheader("Model Predictions and Accuracy:")
        st.write("Here are the predictions and Mean Absolute Error (MAE) from different models:")

        predictions_table = {
            'Model': [],
            'Prediction': [],
            'MAE': []
        }

        data = pd.read_csv("../data/rainfall.csv")

        x = data.iloc[:,:7].values
        y = data.iloc[:,7].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)

        models = [
            ('RF', RandomForestRegressor()),
            ('LR', LinearRegression()),
            ('Ridge', Ridge()),
            ('Lasso', GradientBoostingRegressor())
        ]

        for name, model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mae = mean_squared_error(y_test, y_pred)
            predictions_table['Model'].append(name)
            predictions_table['Prediction'].append(model.predict(new_data_point)[0])
            predictions_table['MAE'].append(mae)

        predictions_df = pd.DataFrame(predictions_table)
        st.table(predictions_df)

def main():
    st.sidebar.title("Sidebar")
    selected_section = st.sidebar.radio("Select Section", [ "Kerala Flood Prediction", "Earthquake Prediction", "Rainfall Prediction"])

    if selected_section == "Kerala Flood Prediction":
        kerala_flood_section()
    if selected_section == "Earthquake Prediction":
        earthquake_section()
    if selected_section == "Rainfall Prediction":
        rainfall_section()

if __name__ == "__main__":
    main()
