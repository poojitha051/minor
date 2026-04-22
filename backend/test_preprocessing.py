import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Create a simple test image (224x224)
test_img = Image.new('RGB', (224, 224), color=(100, 150, 80))
image_bytes = io.BytesIO()
test_img.save(image_bytes, format='PNG')
image_bytes.seek(0)

model_path = 'model/tomato_disease_model.keras'
model = tf.keras.models.load_model(model_path, compile=False)

# Read the image
img = Image.open(io.BytesIO(image_bytes.getvalue())).convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32)

print('Testing different preprocessing methods:')
print('=' * 60)

# Method 1: [0, 1] range (divide by 255)
method1 = img_array / 255.0
pred1 = model.predict(np.expand_dims(method1, axis=0), verbose=0)[0]
print(f'Method 1 [0,1]: Max={np.max(pred1):.4f}, Predicted class: {np.argmax(pred1)}')
print(f'  Top 5: {np.sort(pred1)[-5:][::-1]}')

# Method 2: [-1, 1] range (divide by 127.5 - 1.0)
method2 = (img_array / 127.5) - 1.0
pred2 = model.predict(np.expand_dims(method2, axis=0), verbose=0)[0]
print(f'Method 2 [-1,1]: Max={np.max(pred2):.4f}, Predicted class: {np.argmax(pred2)}')
print(f'  Top 5: {np.sort(pred2)[-5:][::-1]}')

# Method 3: Raw pixels
pred3 = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
print(f'Method 3 Raw: Max={np.max(pred3):.4f}, Predicted class: {np.argmax(pred3)}')
print(f'  Top 5: {np.sort(pred3)[-5:][::-1]}')

# Method 4: ImageNet preprocessing (mean=127.5, std=127.5)
method4 = (img_array - 127.5) / 127.5
pred4 = model.predict(np.expand_dims(method4, axis=0), verbose=0)[0]
print(f'Method 4 Standardized: Max={np.max(pred4):.4f}, Predicted class: {np.argmax(pred4)}')
print(f'  Top 5: {np.sort(pred4)[-5:][::-1]}')
