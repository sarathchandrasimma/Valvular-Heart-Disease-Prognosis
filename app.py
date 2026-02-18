from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator

class HybridModel(BaseEstimator):
    def __init__(self):
        self.rf_clf = RandomForestClassifier(random_state=42)
        self.svm_clf = SVC(probability=True, random_state=42)

    def fit(self, X, y):
        self.rf_clf.fit(X, y)
        self.svm_clf.fit(X, y)
        return self

    def predict(self, X):
        rf_predictions = self.rf_clf.predict_proba(X)[:, 1]
        svm_predictions = self.svm_clf.decision_function(X)
        hybrid_predictions = (rf_predictions + svm_predictions) / 2
        return (hybrid_predictions > 0.5).astype(int)

app = Flask(__name__)

# Paths to new model
MODEL_PATH = os.path.join('model', 'hybrid_model.pkl')

# Load the trained model
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
except FileNotFoundError:
    model = None
    print(f"❌ Error: Model file not found at {MODEL_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if model is None:
            return "Model not loaded. Please check your model file."

        try:
            # Updated fields to match actual/DataSet.csv columns (excluding TARGET)
            fields = [
                'AGE', 'GENDER', 'AORTIC_VALVE', 'LEFT_ATRIUM', 'EDD', 'ESD', 'EF',
                'IVS_D', 'PW_D', 'AORTA', 'I_A_S', 'RVSP', 'RWMA'
            ]
            features = [float(request.form[field]) for field in fields]

            # No scaling, as model was trained on unscaled data
            input_array = np.array([features])

            prediction = model.predict(input_array)[0]

            if prediction == 1:
                result = "✅ Presence of Heart Disease"
                suggestion = (
                    "⚠️ Please consult a cardiologist immediately. "
                    "Maintain a heart-healthy lifestyle: eat low-fat food, avoid smoking, "
                    "engage in moderate physical activity, and monitor blood pressure regularly."
                )
            else:
                result = "🟢 No Heart Disease Detected"
                suggestion = (
                    "🎉 Great job! Keep following a healthy lifestyle. "
                    "Continue regular exercise, eat a balanced diet, and go for periodic checkups."
                )

            return render_template('result.html', result=result, suggestion=suggestion)

        except Exception as e:
            return render_template('result.html', result=f"⚠️ Error: {str(e)}", suggestion="")

    return render_template('predict.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data')
def data():
    return render_template('data.html')

if __name__ == '__main__':
    app.run(debug=True)

