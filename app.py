from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
#load the model
model = pickle.load(open('election_model.pkl','rb'))

@app.route('/')
def home():
    result=''
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST','GET'])
def predict():
    category = int(request.form['CATEGORY'])
    age = int(request.form['AGE'])
    criminal_cases = int(request.form['CRIMINAL CASES'])
    education = int(request.form['EDUCATION'])
    
    # Make prediction and map the result to 'WIN' or 'LOSE'
    result_code = model.predict([[category, age, criminal_cases, education]])[0]
    result = 'WIN' if result_code == 1 else 'LOSE'
    
    return render_template('index.html', result=result, category=category, age=age, criminal_cases=criminal_cases, education=education)
   

if __name__ == '__main__':
    app.run(debug=True)