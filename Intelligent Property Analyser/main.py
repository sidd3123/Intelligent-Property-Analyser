from flask import Flask, render_template, request
import json
from model_py import model
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        inputs = [  request.form['area'],
                    request.form['bedrooms'],
                    request.form['bathrooms'],
                    request.form['stories'],
                    request.form['mainroad'],
                    request.form['guestroom'],
                    request.form['basement'],
                    request.form['hotwaterheating'],
                    request.form['airconditioning'],
                    request.form['parking'],
                    request.form['prefarea'],
                    request.form['furnishingstatus'],
                    request.form['Price_per_sqft']
                    ]
        
        pred=    " ".join([str(item) for item in model.train(inputs)]  ) 

    return render_template('index.html',preduc="preduction : {}".format(pred))
 

if __name__ == '__main__':
    app.run(debug=True)
