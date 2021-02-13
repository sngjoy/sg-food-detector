import os
import re
import io
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from waitress import serve
import inference
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("model/tensorfood.h5")

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/info', methods=['GET'])
def short_description():
    return jsonify(
        {
            "model": "MobileNetv3",
            "input-size": "224x224x3",
            "num-classes": 12,
            "pretrained-on": "ImageNet"
        }
    )

@app.route("/docs", methods=['GET'])
def documentation():
    return render_template('docs.html')
    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # get the file from the request
        img_file = request.files['file']
        app.logger.info(img_file)
        img_bytes = img_file.read()
        img_bytes_arr = io.BytesIO(img_bytes)
        app.logger.info(f'Loading image...')
        img_arr = inference.load_image(img_bytes_arr)
        app.logger.info(f'Making prediction...')
        _, pred, prob = inference.make_prediction(model, img_arr)
        pred = pred.replace('_', ' ').title()
        app.logger.info(f'Predicted {pred} with {round(prob,2)} probability')
        return jsonify(food=pred, probability=str(round(prob,3)))
    else:
        return 'NIL'

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below
    # serve(app, host="0.0.0.0", port=8000)
