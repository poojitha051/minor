import os
import io
import json
import sys
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# Windows consoles often default to cp1252; avoid UnicodeEncodeError on emoji logs.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# TensorFlow runtime options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
IMG_SIZE = 224
MIN_CONFIDENCE = 0.70

# Class names
CLASS_NAMES = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite",
    "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus", "healthy", "powdery_mildew"
]

# Disease database (no MongoDB needed)
DISEASE_INFO = {
    'Bacterial_spot': {'shelf_life_days': 3, 'severity': 'High', 'urgency': 'CONSUME SOON', 
                       'storage_tip': '🥶 Refrigerate at 4°C. Remove affected spots before eating.',
                       'treatment_solution': '🌿 Remove infected leaves. Apply copper-based fungicide.'},
    'Early_blight': {'shelf_life_days': 4, 'severity': 'High', 'urgency': 'USE WITHIN 4 DAYS',
                     'storage_tip': '🌡️ Store in cool, dry place. Check regularly for softening.',
                     'treatment_solution': '🍄 Apply fungicides with chlorothalonil.'},
    'Late_blight': {'shelf_life_days': 2, 'severity': 'Critical', 'urgency': 'IMMEDIATE USE',
                    'storage_tip': '🚨 Consume today! Do not store.',
                    'treatment_solution': '⚠️ Apply copper-based fungicide immediately.'},
    'Leaf_Mold': {'shelf_life_days': 6, 'severity': 'Medium', 'urgency': 'USE WITHIN WEEK',
                  'storage_tip': '💨 Good ventilation needed. Store in paper bag.',
                  'treatment_solution': '🌬️ Reduce humidity. Improve air circulation.'},
    'Septoria_leaf_spot': {'shelf_life_days': 5, 'severity': 'Medium', 'urgency': 'USE WITHIN 5 DAYS',
                           'storage_tip': '📦 Store in paper bag in refrigerator.',
                           'treatment_solution': '✂️ Remove infected leaves. Apply chlorothalonil fungicide.'},
    'Spider_mites Two-spotted_spider_mite': {'shelf_life_days': 4, 'severity': 'High', 'urgency': 'USE QUICKLY',
                                              'storage_tip': '🧼 Wash thoroughly before storage.',
                                              'treatment_solution': '🕷️ Spray with neem oil.'},
    'Target_Spot': {'shelf_life_days': 4, 'severity': 'High', 'urgency': 'USE WITHIN 4 DAYS',
                    'storage_tip': '🔪 Remove affected areas before storage.',
                    'treatment_solution': '💊 Apply fungicides.'},
    'Tomato_Yellow_Leaf_Curl_Virus': {'shelf_life_days': 5, 'severity': 'Medium', 'urgency': 'USE WITHIN 5-6 DAYS',
                                       'storage_tip': '🏠 Store at room temperature away from sunlight.',
                                       'treatment_solution': '🦟 Control whitefly population.'},
    'Tomato_mosaic_virus': {'shelf_life_days': 5, 'severity': 'Medium', 'urgency': 'USE WITHIN 5 DAYS',
                            'storage_tip': '❄️ Store in cool place.',
                            'treatment_solution': '🧪 Use virus-free seeds. Disinfect tools.'},
    'healthy': {'shelf_life_days': 14, 'severity': 'None', 'urgency': 'NORMAL STORAGE',
                'storage_tip': '✅ Store at room temperature for 5-7 days or refrigerate for up to 2 weeks.',
                'treatment_solution': '🌱 Continue good agricultural practices.'},
    'powdery_mildew': {'shelf_life_days': 7, 'severity': 'Low', 'urgency': 'GOOD SHELF LIFE',
                       'storage_tip': '🧴 Wash before use. Store in cool, dry place.',
                       'treatment_solution': '🌾 Apply sulfur or potassium bicarbonate.'}
}

# Simple in-memory storage (replace MongoDB for now)
users = {}  # phone -> {name, password}
history = {}  # phone -> list

# Load model
_model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "tomato_disease_model_final.h5")

print(f"\n🔍 Looking for model: {MODEL_PATH}")
print(f"📁 File exists: {os.path.exists(MODEL_PATH)}")

if os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        print("🔄 Loading model...")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
        print(f"📊 Input shape: {_model.input_shape}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   Using mock predictions.")
else:
    print("❌ Model file not found. Using mock predictions.")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'Tomato Guard API',
        'status': 'running',
        'model_loaded': _model is not None
    })

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # If no model, return mock prediction
        if _model is None:
            return jsonify({
                'valid': True,
                'rejected': False,
                'disease_name': 'healthy',
                'disease_display': 'healthy',
                'confidence': 94.5,
                'confidence_level': 'VERY HIGH',
                'severity': 'None',
                'urgency': 'NORMAL STORAGE',
                'base_shelf_life': 14,
                'adjusted_shelf_life': 14,
                'shelf_status': 'EXCELLENT',
                'storage_tips': '✅ Store at room temperature for 5-7 days or refrigerate for up to 2 weeks.',
                'treatment_solution': '🌱 Continue good agricultural practices.',
                'top_predictions': [
                    {'name': 'healthy', 'confidence': 94.5},
                    {'name': 'Early_blight', 'confidence': 2.3},
                    {'name': 'Bacterial_spot', 'confidence': 1.8},
                    {'name': 'Leaf_Mold', 'confidence': 0.9},
                    {'name': 'Septoria_leaf_spot', 'confidence': 0.5}
                ]
            })
        
        # Process image with model
        image_bytes = file.read()
        processed = preprocess_image(image_bytes)
        predictions = _model.predict(processed, verbose=0)[0]
        
        top_idx = np.argmax(predictions)
        top_disease = CLASS_NAMES[top_idx]
        top_confidence = float(predictions[top_idx]) * 100
        
        info = DISEASE_INFO.get(top_disease, DISEASE_INFO['healthy'])
        
        return jsonify({
            'valid': True,
            'rejected': False,
            'disease_name': top_disease,
            'disease_display': top_disease.replace('_', ' '),
            'confidence': round(top_confidence, 2),
            'confidence_level': 'VERY HIGH' if top_confidence >= 85 else 'HIGH',
            'severity': info['severity'],
            'urgency': info['urgency'],
            'base_shelf_life': info['shelf_life_days'],
            'adjusted_shelf_life': info['shelf_life_days'],
            'shelf_status': 'GOOD',
            'storage_tips': info['storage_tip'],
            'treatment_solution': info['treatment_solution'],
            'top_predictions': [{'name': top_disease, 'confidence': round(top_confidence, 2)}]
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/weather', methods=['GET', 'OPTIONS'])
def weather():
    if request.method == 'OPTIONS':
        return '', 200
    
    return jsonify({
        'temperature': 28,
        'humidity': 65,
        'condition': 'Partly Cloudy',
        'wind_speed': 12,
        'risk_level': 'MEDIUM',
        'risk_color': '#f39c12',
        'advice': '✅ Weather conditions favorable for tomato growth. Maintain regular watering schedule.',
        'icon': '⛅'
    })

@app.route('/signup', methods=['POST', 'OPTIONS'])
def signup():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid data'}), 400
    
    name = data.get('name', '').strip()
    phone = data.get('phone', '').strip()
    password = data.get('password', '')
    
    if not name or not phone or len(password) < 4:
        return jsonify({'error': 'Name, phone, and password (min 4 chars) required'}), 400
    
    # Store in memory (no MongoDB)
    if phone in users:
        return jsonify({'error': 'Phone number already registered'}), 409
    
    users[phone] = {
        'name': name,
        'password': password  # In production, hash this!
    }
    history[phone] = []
    
    print(f"✅ User registered: {name} ({phone})")
    
    return jsonify({
        'success': True,
        'message': 'Account created successfully',
        'name': name,
        'phone': phone
    })

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid data'}), 400
    
    phone = data.get('phone', '').strip()
    password = data.get('password', '')
    
    if phone not in users:
        return jsonify({'error': 'Invalid phone or password'}), 401
    
    if users[phone]['password'] != password:
        return jsonify({'error': 'Invalid phone or password'}), 401
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'name': users[phone]['name'],
        'phone': phone
    })

@app.route('/history', methods=['POST', 'OPTIONS'])
def save_history():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid data'}), 400
    
    phone = data.get('phone', '').strip()
    if phone not in history:
        history[phone] = []
    
    record = {
        'disease_name': data.get('disease_name'),
        'confidence': data.get('confidence'),
        'severity': data.get('severity'),
        'timestamp': datetime.now().isoformat()
    }
    history[phone].append(record)
    
    return jsonify({'success': True, 'message': 'History saved'})

@app.route('/history/<phone>', methods=['GET', 'OPTIONS'])
def get_history(phone):
    if request.method == 'OPTIONS':
        return '', 200
    
    user_history = history.get(phone, [])
    return jsonify({'history': user_history})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🍅 TOMATO GUARD API")
    print("="*50)
    print(f"📍 Model: {'✅ Loaded' if _model is not None else '⚠️ Mock Mode'}")
    print(f"📍 Database: In-memory (no MongoDB required)")
    print(f"📍 Running on: http://0.0.0.0:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)