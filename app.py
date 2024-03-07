# app.py
import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the pre-trained model
cnn = tf.keras.models.load_model('trained_plant_disease_model.h5')
print("Model Loaded Successfully")

# Define disease classes
class_name = ["Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]  # Update with your actual class names

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        print("@@ Input posted = ", file.filename)
        # Save the file to disk
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        
        # Reading and processing the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128,128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        
        # Make predictions
        predictions = cnn.predict(input_arr)
        result_index = np.argmax(predictions)
        model_prediction = class_name[result_index]

        img_str = image_to_base64(image_path)

        return render_template('predict.html', image=img_str, prediction=model_prediction)
    
def image_to_base64(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str


if __name__ == '__main__':
    app.run(debug=True)
