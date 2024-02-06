from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_kerala_flood_models():
    with open('./models/flood_model.pkl', 'rb') as file:
        return pickle.load(file)

def load_earthquake_models():
    with open('./models/earthquake_model.pkl', 'rb') as file:
        return pickle.load(file)

def kerala_flood_prediction(data_input, models):
    data_point = [float(value.strip()) for value in data_input.split(',')]
    new_data_point = np.array(data_point).reshape(1, -1)

    data = pd.read_csv('../data/kerala.csv')
    data['FLOODS'].replace(['YES', 'NO'], [1, 0], inplace=True)
    x = data.iloc[:, 1:14]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    predictions_table = {'Model': [], 'Prediction': [], 'Accuracy': []}

    for model_name, model_type in models.items():
        model = model_type()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        predictions = model.predict(new_data_point)
        predictions_table['Model'].append(model_name)
        predictions_table['Prediction'].append('YES' if predictions[0] == 1 else 'NO')
        predictions_table['Accuracy'].append(accuracy)

    return predictions_table

def earthquake_prediction(data_input, models):
    data_point = [float(value.strip()) for value in data_input.split(',')]
    new_data_point = np.array(data_point).reshape(1, -1)

    data = pd.read_csv("../data/earthquake.csv")
    data = np.array(data)

    X = data[:, 0:-1]
    y = data[:, -1]
    y = y.astype('int')
    X = X.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    predictions_table = {'Model': [], 'Prediction': [], 'Accuracy': []}

    for model_name, model_type in models.items():
        model = model_type()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        predictions = model.predict(new_data_point)

        predictions_table['Model'].append(model_name)
        predictions_table['Prediction'].append(predictions[0])
        predictions_table['Accuracy'].append(accuracy)

    return predictions_table

@app.route('/api/kerala_flood_prediction', methods=['POST'])
def kerala_flood_api():
    data = request.get_json()
    data_input = data['data_input']

    models = load_kerala_flood_models()
    result = kerala_flood_prediction(data_input, models)

    return jsonify(result)

@app.route('/api/earthquake_prediction', methods=['POST'])
def earthquake_api():
    data = request.get_json()
    data_input = data['data_input']

    models = load_earthquake_models()
    result = earthquake_prediction(data_input, models)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
