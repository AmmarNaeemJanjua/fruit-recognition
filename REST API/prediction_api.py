# Importing the necessary libraries
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
from flask_cors import CORS

# Initializing Flask application
app = Flask(__name__)

# Initializing Cross Origin Resource Sharing
CORS(app)

# Loading the image prediction model
model = tf.keras.models.load_model('C:/Users/Ammar Naeem/vscode/REST API/fruits_model.h5')

class_labels = ['Apple', 'Apricot', 'Banana', 'Blueberry', 'Carambola', 'Coconut', 'Corn', 'Dates', 'Eggplant', 'Guava', 'Kiwi', 'Lemon', 'Lime', 'Lychee', 'Mango', 'Orange', 'Pear', 'Pineapple', 'Pitahaya', 'Pomegranate', 'Strawberry', 'Tomato', 'Watermelon']

# Set the threshold for fruit detection
threshold = 0.6

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Extract image data from the request
    image = request.files['image']

    # Process the image using Pillow
    img = Image.open(image)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values

    # Generate predictions using the model
    predictions = model.predict(np.expand_dims(img, axis=0))
    # Get the predicted class index
    predicted_class = np.argmax(predictions, axis=-1)
    # Get the predicted class probability
    predicted_probability = predictions[0][predicted_class.item()]

    if predicted_probability >= threshold:
        # Get the predicted class label
        predicted_class_label = class_labels[predicted_class.item()]
        # Return the predicted class label as a JSON response
        return jsonify({'Predicted Fruit': predicted_class_label})
    else:
        # Return "Not found" if fruit is not detected
        return jsonify({'Predicted Fruit': 'Not found'})

# Run the Flask application
if __name__ == '__main__':
    app.run('0.0.0.0')