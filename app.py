from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('spam_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['email']
    email_vec = vectorizer.transform([email_content])
    prediction = model.predict(email_vec)[0]
    severity = "Low" if prediction == 0 else "High"
    return render_template('result.html', severity=severity)

if __name__ == '__main__':
    app.run(debug=True)
