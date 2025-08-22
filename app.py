from flask import Flask, render_template, request
import pickle
import numpy as np

# Logistic Regression class using gradient descent (the same one used for training)
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            predictions = 1 / (1 + np.exp(-model))  # Sigmoid function
            
            # Compute gradients
            dw = (1 / X.shape[0]) * np.dot(X.T, (predictions - y))
            db = (1 / X.shape[0]) * np.sum(predictions - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = 1 / (1 + np.exp(-model))  # Sigmoid function
        return [1 if p >= 0.5 else 0 for p in predictions]

# Initialize Flask app
app = Flask(__name__)

# Load the trained model from pickle file
with open('student_exam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler (for feature scaling)
with open('scaler.pkl', 'rb') as scaler_file:
    X_min = pickle.load(scaler_file)
    X_max = pickle.load(scaler_file)

# Route for the home page (render the form)
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions (handle form submission)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data from the user input
        features = []
        for feature in ['feature1', 'feature2']:  # Replace with your actual feature names
            features.append(float(request.form[feature]))

        # Normalize the features using Min-Max scaling
        features_normalized = [(f - X_min[i]) / (X_max[i] - X_min[i]) for i, f in enumerate(features)]
        
        # Convert to numpy array for prediction
        features_normalized = np.array(features_normalized).reshape(1, -1)
        
        # Make the prediction
        prediction = model.predict(features_normalized)
        
        # Show result to the user
        result = "Pass" if prediction[0] == 1 else "Fail"
        return render_template('index.html', result=result)

    except Exception as e:
        return str(e)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
