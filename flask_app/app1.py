from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import warnings
import os
import numpy as np

warnings.filterwarnings('ignore')
app = Flask(__name__)

# Load your trained deep learning model
model = tf.keras.models.load_model('potato_models.h5')

# Create a folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':

            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)

            # Load and preprocess the image for inference
            img = image.load_img(file_path, target_size=(256, 256))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img/255.0  # Normalize the image

            # Perform classification using your model
            predicted_classes = model.predict(img)

            # Convert the predicted classes to human-readable labels (e.g., using a class mapping)
            class_mapping = {0: 'late_blight', 1: 'early_blight', 2: 'healthy'}  # Update with your mapping
            predicted_class = class_mapping[np.argmax(predicted_classes)]

            # Remove the temporary file
            os.remove(file_path)

            return f'Predicted class: {predicted_class}'

if __name__ == '__main__':
    app.run(debug=True)
