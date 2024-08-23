from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Ensure the 'uploads' directory exists
upload_folder = 'uploads/'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Load the model without compiling
model = tf.keras.models.load_model('flower_model.h5', compile=False)

# Manually compile the model
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Define the class labels
class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Define the image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).resize((180, 180))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded file to the 'uploads' directory
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        
        # Process the image and predict
        img = preprocess_image(file_path)
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
        predicted_class = class_names[predicted_index]  # Map index to class name
        
        return f'File uploaded successfully. Predicted class: {predicted_class}'
    
    return 'Failed to upload file'

if __name__ == '__main__':
    app.run(debug=True)
