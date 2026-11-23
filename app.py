import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("models/pneumonia_cnn.keras")
IMG_SIZE = (224, 224)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result='No file selected')

    img = Image.open(file).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model.predict(img_array, verbose=0)[0][0])
    label = 'PNEUMONIA' if prob > 0.5 else 'NORMAL'
    result = f'{label} (probability of pneumonia: {prob:.2%})'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)