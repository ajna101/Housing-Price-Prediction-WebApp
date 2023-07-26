from flask import Flask, render_template, request
from ml_model import predict_house_price
from jinja2 import Environment

# Initialize the Flask app
app = Flask(__name__)

# Enable the floatformat filter in Jinja2 environment
app.jinja_env.filters['floatformat'] = lambda value, decimals=2: f"{value:.{decimals}f}"

# Define the route for the home page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle the form submission when the user enters the input features
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the user input from the form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])
    feature5 = float(request.form['feature5'])
    feature6 = float(request.form['feature6'])
    feature7 = float(request.form['feature7'])
    feature8 = float(request.form['feature8'])
    feature9 = float(request.form['feature9'])
    feature10 = float(request.form['feature10'])
    feature11 = float(request.form['feature11'])
    feature12 = float(request.form['feature12'])
    feature13 = float(request.form['feature13'])

    # Call predict_house_price function
    predicted_price = predict_house_price([feature1, feature2, feature3, feature4, feature5, feature6, feature7,
                                           feature8, feature9, feature10, feature11, feature12, feature13])

    # Pass the predicted_price to the result page
    return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
