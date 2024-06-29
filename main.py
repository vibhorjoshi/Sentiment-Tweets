# Deployment using Flask
from flask import Flask, request, render_template
import pickle  # Assuming you're using pickle for the label encoder

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [preprocess_text(message)]
        vect = tfidf.transform(data).toarray()
        prediction = model.predict(vect)
        output = label_encoder.inverse_transform(prediction)
        return render_template('result.html', prediction=output[0])

if __name__ == '__main__':
    app.run(debug=True)