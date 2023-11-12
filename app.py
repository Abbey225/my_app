from flask import Flask, render_template, request
import pickle
from sklearn.linear_model import LogisticRegression
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine

app = Flask(__name__)

# Configure SQLALCHEMY_DATABASE_URI according to your SQL Server setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://stone/test?driver=ODBC Driver 17 for SQL Server'
db = SQLAlchemy(app)

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employed = db.Column(db.Integer)
    bank_balance = db.Column(db.Float)
    annual_salary = db.Column(db.Float)
    predicted_defaulted = db.Column(db.Integer)

# Load the trained model outside the route functions
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Set up database tables
@app.before_first_request
def create_tables():
    db.create_all()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            employed = int(request.form['employed'])
            bank_balance = float(request.form['bank_balance'])
            annual_salary = float(request.form['annual_salary'])

            # Make prediction using the loaded model
            prediction = model.predict([[employed, bank_balance, annual_salary]])

            # Save the prediction result to the database
            result_entry = PredictionResult(
                employed=employed,
                bank_balance=bank_balance,
                annual_salary=annual_salary,
                predicted_defaulted=int(prediction[0])
            )
            db.session.add(result_entry)
            db.session.commit()

            return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        # Handle exceptions, log them, or return an error page
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
