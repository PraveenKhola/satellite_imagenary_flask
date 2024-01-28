import random

from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO


import os
import cv2
import numpy as np
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import matplotlib.pyplot as plt


app = Flask(__name__)


weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


from tensorflow.keras.models import load_model
model = load_model("satellite_imagenary_unet.hdf5",custom_objects={'dice_loss_plus_1focal_loss': total_loss})

model_2 = load_model("poverty_prediction.hdf5")

@app.route("/")
def home():
    return jsonify({'result':"Not the valid "})


@app.route("/landscape_detection", methods=["Post"])
def process_image():
    # Check if the post request has the file part
    # print(request)
    # file = request.files['file']
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part',"request_body":request.args})
    #
    file = request.files['file']
    #
    # # If the user does not select a file, the browser submits an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image file
    image = Image.open(file)
    image = np.array(image)
    # image = plt.imread('download.jpeg')
    image = cv2.resize(image, (256, 256))
    predicted_image = np.argmax(model.predict(np.expand_dims(image, axis=0)), axis=3)
    predicted_image = np.squeeze(predicted_image).astype(np.uint8)
    predicted_image = Image.fromarray(predicted_image)

    output_buffer = BytesIO()
    predicted_image.save(output_buffer, format='PNG')

    # Convert BytesIO buffer to bytes
    output_image_bytes = output_buffer.getvalue()

    # Create a response with the processed image
    response = jsonify({'result': 'success', 'processed_image': output_image_bytes.decode('latin1')})
    response.headers['Content-Type'] = 'application/json'

    return response


@app.route("/poverty_prediction",methods=["POST"])
def predict_poverty():

    file = request.files["file"]

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image,axis=0)
    random_number = random.random()
    prediction = model_2.predict(image)
    prediction = prediction[0][2]*5 - random_number


    response = jsonify({'result': 'success', 'poverty_predicted': prediction})
    response.headers['Content-Type'] = 'application/json'

    return response


app.run(debug=True)
