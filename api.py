# import the necessary modules
from tensorflow.keras.applications import ResNet50  # pre-built CNN Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import os
from flask import request
import requests

# Create Flask application and initialize Keras model
app = flask.Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './model/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')


# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# print('Model loaded. Check http://127.0.0.1:5000/')
def read_image_from_file(filename):
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploaded', filename)
    return file_path

def read_image_from_url(url):
    img = Image.open(requests.get(url, stream=True).raw)
    return img

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # target_size must agree with what the trained model expects!!

    # Preprocessing the image
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    predict = model.predict(images, batch_size=1)
    return predict

# Every ML/DL model has a specific format
# of taking input. Before we can predict on
# the input image, we first need to preprocess it.

# Now, we can predict the results.
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' in request.files:
        image_request = request.files['image'].read
        img = read_image_from_file(image_request)
    elif 'url' in request.json:
        image_url = request.json['url']
        img = read_image_from_url(image_url)
    else:
        return 0
    predict = model_predict(img, model)

    if predict[0][0] == 1:
        return make_response(jsonify({'message': 'Normal'}), 200)
    else:
        return make_response(jsonifY({'message': 'Pneumonia'}), 200)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
