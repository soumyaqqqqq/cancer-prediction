import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'brain_tumor_model.keras')
IMG_SIZE = 150
CLASS_NAMES = ['class_0', 'class_1', 'class_2', 'class_3']  # Replace with real class names

# Load model
model = load_model(MODEL_PATH)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Save uploaded image to training directory
        label = request.form.get('label', 'unlabeled')  # Optionally get label from form
        save_dir = os.path.join(os.path.dirname(__file__), '../data/brain-data/Training', label)
        os.makedirs(save_dir, exist_ok=True)
        filename = secure_filename(file.filename)
        save_path = os.path.join(save_dir, filename)
        file.seek(0)
        file.save(save_path)

        # Predict as before
        img = image.load_img(save_path, target_size=(IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        # Retrain model after saving new image
        retrain_model()

        return jsonify({
            'predicted_class': CLASS_NAMES[pred_class],
            'confidence': confidence,
            'all_confidences': {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))},
            'retrained': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add retrain_model function
def retrain_model():
    train_dir = os.path.join(os.path.dirname(__file__), '../data/brain-data/Training')
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )
    # Recompile model (if needed)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=1)  # Use 1 epoch for demo; increase as needed
    model.save(MODEL_PATH)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
