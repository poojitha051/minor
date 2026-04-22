import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Create test tomato leaf (greenish image)
test_img = Image.new('RGB', (224, 224), color=(50, 120, 30))
image_bytes = io.BytesIO()
test_img.save(image_bytes, format='PNG')
image_bytes.seek(0)

model_path = 'model/tomato_disease_model.keras'
model = tf.keras.models.load_model(model_path, compile=False)

# Read and preprocess
img = Image.open(io.BytesIO(image_bytes.getvalue())).convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32)

# Test with [0, 1] normalization
img_normalized = img_array / 255.0
pred = model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0]

print('Prediction scores:', pred)
print('Max confidence:', np.max(pred))
print('Top class index:', np.argmax(pred))
print('Top 5 predictions:')
top_indices = np.argsort(pred)[::-1][:5]
for i, idx in enumerate(top_indices):
    print(f'  {i+1}. Class {idx}: {pred[idx]:.4f}')
