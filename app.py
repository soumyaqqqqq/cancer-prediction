import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Configuration
MODEL_PATH = os.path.join('model', 'brain_tumor_model.keras')
IMG_SIZE = 150
CLASS_NAMES = ['class_0', 'class_1', 'class_2', 'class_3']  # Update with your actual class names

# Load model
model = load_model(MODEL_PATH)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    img = image.load_img(file, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_class = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    return jsonify({
        'predicted_class': CLASS_NAMES[pred_class],
        'confidence': confidence,
        'all_confidences': {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
