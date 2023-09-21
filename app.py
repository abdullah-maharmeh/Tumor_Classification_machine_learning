from flask import Flask, flash, request, redirect, url_for, render_template
import numpy as np
import urllib.request
import cv2
import pickle
# from flask_wtf import FlaskForm
# from wtforms import FileField , SubmitField
from werkzeug.utils import secure_filename
import os 

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/files/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('home.html')
 

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Process the uploaded image using the pipeline
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = cv2.imread(image_path)
        filtered_image = cv2.pyrMeanShiftFiltering(image, 21, 51)

        resized_image = cv2.resize(filtered_image, (180, 180))
        resized_to_desplay = cv2.resize(image,(250,250))
        resized_filename = 'resized_' + filename  # Rename to avoid overwriting original image
        resized_path = os.path.join(app.config['UPLOAD_FOLDER'], resized_filename)
        cv2.imwrite(resized_path, resized_to_desplay)

        normalized_image = resized_image / 255
        image_features = normalized_image.reshape(1, -1)
        
        pca = pickle.load(open('pca.sav', 'rb'))
        image_pca = pca.transform(image_features)
        
        model = pickle.load(open('model_SVM.sav', 'rb'))
        result = model.predict(image_pca)
        
        if result == 'Benign':
            text = 'This image contains Benign Tumor'
        elif result =='Malignant':
            text = 'This image contains Malignant Tumor'
        else :
            text = 'This is normal image'
        
        # flash('Image successfully uploaded and processed. Prediction: ' + result_text)
        
        return render_template('home.html', filename=resized_filename ,result = text)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    
@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='files/' + filename), code=301)


if __name__ == "__main__":
    app.run()
