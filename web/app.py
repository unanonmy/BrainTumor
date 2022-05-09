from flask import Flask, render_template, request
import keras
import cv2
import numpy as np
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/prediction')
def prediction():
    return render_template("prediction.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        file_name = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],file_name)
        f.save(file_path)

    md = keras.models.load_model(r"C:\Users\DELL\Desktop\dataset\web\model.h5")
    x = cv2.imread(file_path,0)
    x = np.expand_dims(x, axis=0)
    res = md.predict(x)
    response = np.argmax(res)
    class_type = ["meningioma","glioma","pituitary"]
    result = class_type[response]
    os.remove(file_path)
    return render_template(result+".html")

if __name__ == '__main__':
    app.run(debug=True, port=5500)