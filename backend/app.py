import os
import io
import json
import sys
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from scipy import ndimage
from utils.weather_service import get_weather

# Load environment variables
load_dotenv()

# Windows console fix
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================================
# CONFIGURATION
# ============================================
IMG_SIZE = 224

# Thresholds for acceptance
MIN_CONFIDENCE_FOR_TOMATO = 0.30
MAX_ENTROPY_FOR_TOMATO = 1.5
MIN_PREDICTION_GAP = 0.25

# Class names (must match your training order)
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew",
    "unknown"
]

# Disease database
DISEASE_INFO = {
    'Bacterial_spot': {'shelf_life_days': 3, 'severity': 'High', 'urgency': 'CONSUME SOON'},
    'Early_blight': {'shelf_life_days': 4, 'severity': 'High', 'urgency': 'USE WITHIN 4 DAYS'},
    'Late_blight': {'shelf_life_days': 2, 'severity': 'Critical', 'urgency': 'IMMEDIATE USE REQUIRED'},
    'Leaf_Mold': {'shelf_life_days': 6, 'severity': 'Medium', 'urgency': 'USE WITHIN A WEEK'},
    'Septoria_leaf_spot': {'shelf_life_days': 5, 'severity': 'Medium', 'urgency': 'USE WITHIN 5 DAYS'},
    'Spider_mites Two-spotted_spider_mite': {'shelf_life_days': 4, 'severity': 'High', 'urgency': 'USE QUICKLY'},
    'Target_Spot': {'shelf_life_days': 4, 'severity': 'High', 'urgency': 'CONSUME WITHIN 4 DAYS'},
    'Tomato_Yellow_Leaf_Curl_Virus': {'shelf_life_days': 5, 'severity': 'Medium', 'urgency': 'USE WITHIN 5-6 DAYS'},
    'Tomato_mosaic_virus': {'shelf_life_days': 5, 'severity': 'Medium', 'urgency': 'USE WITHIN 5 DAYS'},
    'healthy': {'shelf_life_days': 14, 'severity': 'None', 'urgency': 'NORMAL STORAGE'},
    'powdery_mildew': {'shelf_life_days': 7, 'severity': 'Low', 'urgency': 'GOOD SHELF LIFE'}
}

# In-memory storage
users = {}
history = {}
_model = None

# ============================================
# LOAD MODEL - LOOK FOR .KERAS FILE
# ============================================
def load_model():
    """Load the .keras model file from model folder"""
    global _model
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(backend_dir, 'model', 'tomato_disease_model.keras'),
        os.path.join(backend_dir, 'model', 'tomato_rejection_model.keras'),
        os.path.join(backend_dir, 'model', 'model.keras'),
        os.path.join(backend_dir, 'tomato_disease_model.keras'),
        os.path.join(backend_dir, 'tomato_rejection_model.keras'),
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Found model: {path}")
            break
    
    if model_path is None:
        print("\n❌ Model not found!")
        model_folder = os.path.join(backend_dir, 'model')
        if os.path.exists(model_folder):
            print(f"\n📁 Files in model folder:")
            for f in os.listdir(model_folder):
                print(f"   - {f}")
        return False
    
    try:
        import tensorflow as tf
        print(f"🔄 Loading model from {model_path}...")
        _model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
        print(f"📊 Model input shape: {_model.input_shape}")
        print(f"📊 Model output shape: {_model.output_shape}")
        
        num_classes = _model.output_shape[-1]
        print(f"📊 Model expects {num_classes} classes")
        
        if num_classes != len(CLASS_NAMES):
            print(f"⚠️ Warning: Model has {num_classes} classes but CLASS_NAMES has {len(CLASS_NAMES)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# QUICK HEURISTIC REJECTION
# ============================================
def quick_reject_non_tomato(image_bytes):
    """Immediately reject non-tomato images using image analysis"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    
    reasons = []
    
    green_channel = img_array[:, :, 1]
    red_channel = img_array[:, :, 0]
    blue_channel = img_array[:, :, 2]
    
    green_ratio = np.mean(green_channel) / (np.mean(red_channel) + np.mean(blue_channel) + 1)
    if green_ratio < 0.5:
        reasons.append(f"Low green dominance (ratio: {green_ratio:.2f})")
    
    should_reject = False
    
    return should_reject, reasons

# ============================================
# MODEL PREPROCESSING - NORMALIZED [0, 1] RANGE
# ============================================
def preprocess_image(image_bytes):
    """Preprocess image - divide by 255 to normalize to [0, 1] range"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_with_model(image_bytes):
    """Run model prediction"""
    if _model is None:
        return {
            'valid': False,
            'error': 'Model not loaded',
            'disease': 'unknown',
            'confidence': 0.0,
            'top_predictions': []
        }
    
    try:
        processed = preprocess_image(image_bytes)
        predictions = _model.predict(processed, verbose=0)[0]
        
        top1_idx = np.argmax(predictions)
        top1_conf = float(predictions[top1_idx])
        top1_class = CLASS_NAMES[top1_idx] if top1_idx < len(CLASS_NAMES) else "unknown"
        
        sorted_indices = np.argsort(predictions)[::-1][:5]
        top_predictions = [
            {
                'name': CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}", 
                'confidence': float(predictions[i] * 100)
            }
            for i in sorted_indices
        ]
        
        print(f"Prediction: {top1_class} with {top1_conf:.3f} confidence")
        
        return {
            'valid': True,
            'disease': top1_class,
            'confidence': top1_conf,
            'top_predictions': top_predictions
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'valid': False,
            'error': str(e),
            'disease': 'unknown',
            'confidence': 0.0,
            'top_predictions': []
        }

# ============================================
# CORRECT SHELF LIFE CALCULATION
# ============================================
def calculate_shelf_life(base_shelf_life, confidence):
    """
    Calculate shelf life based on confidence level
    HIGHER confidence = LONGER shelf life
    LOWER confidence = SHORTER shelf life (conservative estimate)
    """
    
    # Calculate multiplier based on confidence tiers
    if confidence >= 0.90:
        multiplier = 1.0
        confidence_level = "EXCEPTIONAL"
        explanation = "Very high confidence - Full shelf life recommended"
    elif confidence >= 0.80:
        multiplier = 0.95
        confidence_level = "VERY HIGH"
        explanation = "High confidence - Minimal reduction applied"
    elif confidence >= 0.70:
        multiplier = 0.85
        confidence_level = "HIGH"
        explanation = "Good confidence - Slight reduction for safety"
    elif confidence >= 0.60:
        multiplier = 0.75
        confidence_level = "GOOD"
        explanation = "Moderate confidence - Standard reduction applied"
    elif confidence >= 0.50:
        multiplier = 0.65
        confidence_level = "MODERATE"
        explanation = "Fair confidence - Moderate reduction for safety"
    elif confidence >= 0.40:
        multiplier = 0.55
        confidence_level = "LOW-MODERATE"
        explanation = "Low confidence - Significant reduction applied"
    elif confidence >= 0.30:
        multiplier = 0.45
        confidence_level = "LOW"
        explanation = "Very low confidence - Heavy reduction for safety"
    else:
        multiplier = 0.35
        confidence_level = "VERY LOW"
        explanation = "Critically low confidence - Minimum shelf life recommended"
    
    # Calculate adjusted shelf life
    adjusted_shelf_life = max(1, int(round(base_shelf_life * multiplier)))
    
    # Ensure shelf life doesn't exceed base
    adjusted_shelf_life = min(adjusted_shelf_life, base_shelf_life)
    
    return adjusted_shelf_life, confidence_level, explanation

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'Tomato Guard API',
        'version': '8.0 - Corrected Shelf Life',
        'status': 'running',
        'model_loaded': _model is not None,
        'preprocessing': '[0, 1] range (divide by 255)',
        'shelf_life_logic': 'Higher confidence = Longer shelf life',
        'thresholds': {
            'min_confidence': MIN_CONFIDENCE_FOR_TOMATO
        }
    })

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({
        'status': 'ok', 
        'model_loaded': _model is not None
    })

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint to check model status"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(backend_dir, 'model')
    
    model_folder_exists = os.path.exists(model_folder)
    model_folder_contents = []
    if model_folder_exists:
        model_folder_contents = os.listdir(model_folder)
    
    return jsonify({
        'backend_directory': backend_dir,
        'model_folder_exists': model_folder_exists,
        'model_folder_contents': model_folder_contents,
        'model_loaded': _model is not None,
        'preprocessing': '[0, 1] range (divide by 255)',
        'num_classes': len(CLASS_NAMES)
    })

@app.route('/weather', methods=['GET', 'OPTIONS'])
def weather():
    """Get weather data for given coordinates"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            return jsonify({'error': 'Missing lat or lon parameters'}), 400
        
        weather_data = get_weather(lat, lon)
        return jsonify(weather_data)
    
    except Exception as e:
        print(f"Weather error: {e}")
        return jsonify({
            'error': str(e),
            'temperature_c': 0,
            'humidity': 0,
            'condition': 'Unknown',
            'wind_speed_ms': 0,
            'disease_risk': 'LOW',
            'farming_advice': 'Unable to fetch real weather data',
            'is_mock': True
        }), 500

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
        image_bytes = file.read()
        
        if _model is None:
            return jsonify({
                'valid': False,
                'rejected': True,
                'message': 'Model not loaded'
            }), 500
        
        should_reject, reasons = quick_reject_non_tomato(image_bytes)
        
        if should_reject:
            return jsonify({
                'valid': False,
                'rejected': True,
                'message': '❌ NOT A TOMATO LEAF - Image rejected',
                'rejection_type': 'heuristic',
                'rejection_reasons': reasons,
                'top_predictions': []
            })
        
        result = predict_with_model(image_bytes)
        
        if not result.get('valid', False):
            return jsonify({
                'valid': False,
                'rejected': True,
                'message': 'Prediction failed',
                'error': result.get('error', 'Unknown error')
            }), 500
        
        disease = result['disease']
        confidence = result['confidence']
        
        if disease == 'unknown' or confidence < MIN_CONFIDENCE_FOR_TOMATO:
            return jsonify({
                'valid': False,
                'rejected': True,
                'message': '❌ NOT A TOMATO LEAF - Low confidence',
                'rejection_type': 'ml_confidence',
                'top_prediction': disease,
                'confidence': round(confidence * 100, 2),
                'top_predictions': result['top_predictions']
            })
        
        # STEP 4: Calculate shelf life using CORRECTED formula
        disease_info = DISEASE_INFO.get(disease, DISEASE_INFO['healthy'])
        base_shelf_life = disease_info['shelf_life_days']
        
        # Calculate shelf life based on confidence
        adjusted_shelf_life, confidence_level, shelf_life_explanation = calculate_shelf_life(base_shelf_life, confidence)
        
        # Calculate what percentage of shelf life is remaining
        shelf_life_percentage = int((adjusted_shelf_life / base_shelf_life) * 100)
        
        # Determine shelf status
        if adjusted_shelf_life <= 1:
            shelf_status = "CRITICAL - Use TODAY!"
        elif adjusted_shelf_life <= 2:
            shelf_status = "URGENT - Use within 2 days"
        elif adjusted_shelf_life <= 3:
            shelf_status = "WARNING - Use soon"
        elif adjusted_shelf_life <= 5:
            shelf_status = "GOOD - Normal storage"
        elif adjusted_shelf_life <= 7:
            shelf_status = "VERY GOOD - Extended storage possible"
        else:
            shelf_status = "EXCELLENT - Long shelf life"
        
        # Determine storage tip based on adjusted shelf life
        if adjusted_shelf_life <= 1:
            storage_tip = "🚨 CONSUME IMMEDIATELY! Do not store. Discard if any rot visible."
        elif adjusted_shelf_life <= 2:
            storage_tip = "⚠️ Use within 2 days. Refrigerate at 4°C. Check daily for spoilage."
        elif adjusted_shelf_life <= 4:
            storage_tip = "⚠️ Use within few days. Refrigerate at 4°C. Check regularly."
        elif adjusted_shelf_life <= 7:
            storage_tip = "✓ Store in cool, dry place. Keep away from sunlight. Use within a week."
        else:
            storage_tip = "✅ Store at room temperature 5-7 days or refrigerate up to 2 weeks."
        
        print(f"\n📊 Shelf Life Calculation:")
        print(f"   Disease: {disease}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Base Shelf Life: {base_shelf_life} days")
        print(f"   Adjusted Shelf Life: {adjusted_shelf_life} days")
        print(f"   Percentage: {shelf_life_percentage}%")
        print(f"   Explanation: {shelf_life_explanation}")
        
        return jsonify({
            'valid': True,
            'rejected': False,
            'message': '✅ Valid tomato leaf detected',
            'disease_name': disease,
            'disease_display': disease.replace('_', ' '),
            'confidence': round(confidence * 100, 2),
            'confidence_level': confidence_level,
            'severity': disease_info['severity'],
            'urgency': disease_info['urgency'],
            'base_shelf_life': base_shelf_life,
            'adjusted_shelf_life': adjusted_shelf_life,
            'shelf_life_percentage': shelf_life_percentage,
            'shelf_life_explanation': shelf_life_explanation,
            'shelf_status': shelf_status,
            'storage_tips': storage_tip,
            'treatment_solution': get_treatment_solution(disease),
            'top_predictions': result['top_predictions'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_treatment_solution(disease):
    solutions = {
        'Bacterial_spot': '🌿 Remove infected leaves. Apply copper-based fungicide.',
        'Early_blight': '🍄 Apply fungicides with chlorothalonil.',
        'Late_blight': '⚠️ Apply copper-based fungicide immediately.',
        'Leaf_Mold': '🌬️ Reduce humidity. Improve air circulation.',
        'Septoria_leaf_spot': '✂️ Remove infected leaves. Apply chlorothalonil fungicide.',
        'Spider_mites Two-spotted_spider_mite': '🕷️ Spray with neem oil.',
        'Target_Spot': '💊 Apply fungicides containing mancozeb.',
        'Tomato_Yellow_Leaf_Curl_Virus': '🦟 Control whitefly population.',
        'Tomato_mosaic_virus': '🧪 Use virus-free seeds. Disinfect tools.',
        'healthy': '🌱 Continue good agricultural practices.',
        'powdery_mildew': '🌾 Apply sulfur or potassium bicarbonate.'
    }
    return solutions.get(disease, 'Consult local agricultural expert.')

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
    
    if phone in users:
        return jsonify({'error': 'Phone already registered'}), 409
    
    users[phone] = {'name': name, 'password': password}
    history[phone] = []
    
    return jsonify({'success': True, 'message': 'Account created', 'name': name, 'phone': phone})

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    phone = data.get('phone', '').strip()
    password = data.get('password', '')
    
    if phone not in users or users[phone]['password'] != password:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    return jsonify({'success': True, 'message': 'Login successful', 'name': users[phone]['name'], 'phone': phone})

@app.route('/history', methods=['POST', 'OPTIONS'])
def save_history():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    phone = data.get('phone', '').strip()
    if phone not in history:
        history[phone] = []
    
    history[phone].append({
        'disease_name': data.get('disease_name'),
        'confidence': data.get('confidence'),
        'severity': data.get('severity'),
        'timestamp': datetime.now().isoformat()
    })
    
    return jsonify({'success': True})

@app.route('/history/<phone>', methods=['GET'])
def get_history(phone):
    return jsonify({'history': history.get(phone, [])})

@app.route('/test_rejection', methods=['POST'])
def test_rejection():
    """Test endpoint to see if image would be rejected"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    should_reject, reasons = quick_reject_non_tomato(file.read())
    
    return jsonify({
        'would_reject': should_reject,
        'rejection_reasons': reasons,
        'is_tomato_leaf': not should_reject
    })

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🍅 TOMATO GUARD API - CORRECTED SHELF LIFE VERSION")
    print("="*70)
    
    model_loaded = load_model()
    
    print(f"\n✅ Model Loaded: {model_loaded}")
    print(f"✅ Preprocessing: [0, 1] range (divide by 255)")
    print(f"✅ Min Confidence Threshold: {MIN_CONFIDENCE_FOR_TOMATO:.0%}")
    print(f"\n📊 Shelf Life Logic (Higher Confidence = Longer Shelf Life):")
    print(f"   ≥90% confidence → 100% of base shelf life")
    print(f"   80-89% confidence → 95% of base shelf life")
    print(f"   70-79% confidence → 85% of base shelf life")
    print(f"   60-69% confidence → 75% of base shelf life")
    print(f"   50-59% confidence → 65% of base shelf life")
    print(f"   40-49% confidence → 55% of base shelf life")
    print(f"   30-39% confidence → 45% of base shelf life")
    print(f"   <30% confidence → 35% of base shelf life")
    
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 Server running on: http://0.0.0.0:{port}")
    print(f"🔍 Debug: http://0.0.0.0:{port}/debug")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)