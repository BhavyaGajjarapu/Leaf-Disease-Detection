import cv2
import os
from werkzeug.utils import secure_filename
from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

model = load_model('best_model.h5')

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prediction(path):
    ref = {0: 'Apple___Apple_scab',1: 'Apple___Black_rot',2: 'Apple___Cedar_apple_rust',3: 'Apple___healthy',4: 'Blueberry___healthy',5: 'Cherry_(including_sour)___Powdery_mildew',6: 'Cherry_(including_sour)___healthy',7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',8: 'Corn_(maize)___Common_rust_',9: 'Corn_(maize)___Northern_Leaf_Blight',10: 'Corn_(maize)___healthy',11: 'Grape___Black_rot',12: 'Grape___Esca_(Black_Measles)',13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',14: 'Grape___healthy',15: 'Orange___Haunglongbing_(Citrus_greening)',16: 'Peach___Bacterial_spot',17: 'Peach___healthy',18: 'Pepper,_bell___Bacterial_spot',19: 'Pepper,_bell___healthy',20: 'Potato___Early_blight',21: 'Potato___Late_blight',22: 'Potato___healthy',23: 'Raspberry___healthy',24: 'Soybean___healthy',25: 'Squash___Powdery_mildew',26: 'Strawberry___Leaf_scorch',27: 'Strawberry___healthy',28: 'Tomato___Bacterial_spot',29: 'Tomato___Early_blight',30: 'Tomato___Late_blight',31: 'Tomato___Leaf_Mold',32: 'Tomato___Septoria_leaf_spot',33: 'Tomato___Spider_mites Two-spotted_spider_mite',34: 'Tomato___Target_Spot',35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',36: 'Tomato___Tomato_mosaic_virus',37: 'Tomato___healthy'}
    img = load_img(path,target_size=(256,256))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    pred = np.argmax(model.predict(img))
    return ref[pred]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        pred = prediction(UPLOAD_FOLDER+'/'+filename)
        return render_template('home.html',org_img_name=filename,prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)