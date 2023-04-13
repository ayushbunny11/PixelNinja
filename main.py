from __future__ import division, print_function
import io
# coding=utf-8
import os
import numpy as np
import keras.backend as K
from conf import myConfig as config
import argparse
import cv2
from pathlib import Path
import base64
from PIL import Image
# TKINTER FOR DOWNLOADING
import tkinter as tk
from tkinter import filedialog

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from tensorflow import keras

# Define a flask app
app = Flask(__name__)


# createModel, loadWeights
# this is required for loading a keras-model created with custom-loss
def custom_loss(y_true, y_pred):
    diff = abs(y_true-y_pred)
    res = (diff)/(config.batch_size)
    return res


model = load_model(
    "./models/B5.h5", custom_objects={'custom_loss': custom_loss})
print('Model loaded. Start serving...')


print('Model loaded. Check http://127.0.0.1:5000/')


def ENL(img):
    mean = np.average(img)
    std = np.std(img)
    ENL1 = (mean*mean)/(std*std)
    return ENL1


def model_predict(path, model):

    lenth = 1
    sumPSNR = 0
    sumSSIM = 0
    psnr_val = np.empty(lenth)
    ssim_val = np.empty(lenth)
    np.random.seed(seed=0)  # for reproducibility
    img1 = (cv2.imread(str(path), 0))/255.
    z = np.squeeze(model.predict(np.expand_dims(img1, axis=0)))
    enl = ENL(z)
    print(enl)
    # cv2.imwrite("uploads/"+str(i+1)+"_Original.png",255.*img1)
    return 255.*z


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)

        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))

        f.save(file_path)
        path = str(file_path)

        # Make prediction
        upload.preds = model_predict(path, model)
        return "Saved"

    else:
        return render_template('index.html')


# ----------------------------------------------------DOWNLOADING THE FILE-------------------------------------------------


@app.route('/download', methods=['GET', 'POST'])
def download():
    if request.method == 'GET':

        # Get the list of all files in the folder using os.listdir()
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'downloads')
        files = os.listdir(file_path)

        # Use a list comprehension to count the number of files in the list
        num_files = len(
            [f for f in files if os.path.isfile(os.path.join(file_path, f))])

        # For storing the image with different names so that it doesnt collide
        ar = ["downloads/imageDemo", str(num_files+1), ".jpeg"]

        success = cv2.imwrite(''.join(ar), upload.preds)
        if success:
            return send_file(''.join(ar), mimetype='image/jpeg')
        else:
            return ""

    else:
        return "Error!!"


@app.route('/processed-image', methods=['GET'])
def get_processed_image():

    print(upload.preds)

    processed_image = Image.fromarray(np.uint8(upload.preds))
    # Convert the image to a bytes buffer and serve it to the client
    img_buffer = io.BytesIO()
    processed_image.save(img_buffer, format='JPEG')
    encoded_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return jsonify({'imageData': encoded_image})


if __name__ == '__main__':
    app.run()
