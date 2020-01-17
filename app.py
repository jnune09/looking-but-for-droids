# coding=utf-8
import sys
import os
import io
import glob
import re
import numpy as np
from PIL import Image

# Tensorflow
import tensorflow as tf

# Keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import mobilenet

# Flask utils
from flask import Flask, jsonify, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/mobilenet_classifier.h5'
model = None


def get_model():
	global model
	model = load_model(MODEL_PATH)
	global graph
	graph = tf.get_default_graph()


def prepare_image(image, target):
	if image.mode != "RGB":
		image = image.convert("RGB")

	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	mobilenet.preprocess_input(image)

	return image


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image, target=(224, 224))

        with graph.as_default():
            prediction = model.predict(image).tolist()
            
        if prediction[0][0] > prediction[0][1]:
            result = 'c3po'
        else:
            result = 'r2d2'

        return result
    return None

@app.route("/api/predict", methods=["POST"])
def predict():
	if request.files.get("image"):
		image = request.files["image"].read()
		image = Image.open(io.BytesIO(image))
		image = prepare_image(image, target=(224, 224))

		with graph.as_default():
			prediction = model.predict(image).tolist()

		response = {
			'prediction': {
				'c3po': prediction[0][0],
				'r2d2': prediction[0][1]
			}
		}
		return jsonify(response)

if __name__ == '__main__':
    get_model()

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5005), app)
    http_server.serve_forever()
