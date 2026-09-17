from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        # Convert input data to DataFrame
        query_df = pd.DataFrame(json_)
        
        # Ensure the column order is correct if necessary, based on training data
        # Assuming 'Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
        # are the expected feature columns in that order
        expected_columns = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        query_df = query_df[expected_columns]

        prediction = model.predict(query_df)
        return jsonify({'prediction': list(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For local development, you can run:
    # app.run(debug=True, host='0.0.0.0', port=5000)
    # For deployment, a production-ready server like Gunicorn should be used.
    print("To run the app, save this code as app.py and execute it using 'python app.py' or a WSGI server like Gunicorn.")
