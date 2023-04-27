from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.optimizers import Adam
from loss import total_loss
from flask import Flask, request, render_template, send_file, make_response
from flask_restful import Resource, Api, reqparse
import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)
api = Api(app)
port = 5000

# Model saved with Keras model.save()
MODEL_PATH ='Models/9/9_g_model.h5'

if sys.argv.__len__() > 1:
    port = sys.argv[1]
print("You said port is : {} ".format(port))

class HelloWorld(Resource):
    def get(self):
        return make_response(render_template('index.html'))
        # return {'hello': 'world Port : {} '.format(port)}

class ImageProcessor(Resource):
    def post(self):
        # Get the uploaded file from the request
        file = request.files['file']

        img = load_img(file, target_size=(256, 256))
        img_array = np.array(img)
        norm_img = (img_array.copy() - 127.5) / 127.5
        # Load trained model
        model = load_model(MODEL_PATH, custom_objects={'InstanceNormalization': InstanceNormalization})

        g_img = model.predict(np.expand_dims(norm_img, 0))[0]
        g_img = g_img * 127.5 + 127.5
        g_img = cv2.resize(g_img, (200, 250))

        output_img = g_img
        output_img = Image.fromarray(output_img.astype(np.uint8))

        # Save the image to a BytesIO buffer
        buffer = io.BytesIO()
        output_img.save(buffer, format='PNG')
        buffer.seek(0)

        # Return the image as a Flask response
        response = make_response(buffer.getvalue())
        response.headers.set('Content-Type', 'image/png')
        response.headers.set('Content-Disposition', 'attachment', filename='image.png')
        return response

        # return {"Success":"type1"}


api.add_resource(HelloWorld, '/')
api.add_resource(ImageProcessor, '/process_image')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)
