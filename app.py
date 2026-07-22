from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================
# Load TFLite Model Once (at startup) — Lightweight
# ============================================
interpreter = tf.lite.Interpreter(model_path="skin_tone_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["deep", "medium", "fair"]

# ============================================
# Gemini Client
# ============================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
google_search_tool = types.Tool(google_search=types.GoogleSearch())

# ============================================
# Face Detection — Lightweight Haar Cascade (No MTCNN, No TensorFlow overhead)
# ============================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    cv2.imwrite(output_path, face_crop)
    return output_path

# ============================================
# Predict Skin Tone (TFLite Interpreter)
# ============================================
def predict_skin_tone(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    probabilities = {
        class_names[i]: round(float(prediction[i]) * 100, 2)
        for i in range(len(class_names))
    }

    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]

    return predicted_class, confidence, probabilities

# ============================================
# Get Clothing + Accessory Suggestions
# ============================================
def get_clothing_suggestions(probabilities, clothing_type, size, occasion):
    prob_str = ", ".join([f"{k}: {v}%" for k, v in probabilities.items()])

    prompt = f"""
You are an expert men's fashion stylist and color analyst with real-time Google Search access.

A machine learning model analyzed a person's face and produced these skin tone probabilities:
{prob_str}

Based on established color theory and skin undertone matching principles, decide for 
yourself which colors would best complement this specific skin tone profile.

Then:
1. Search the live web to find REAL, currently listed {clothing_type} products in size {size}, 
   suitable for a {occasion} occasion, in the colors you determined suit this skin tone.
2. Also give brief, minimal accessory color guidance — one short recommended color each 
   for: watch strap/case, shoes, and belt.

Search across Myntra, Ajio, Flipkart, Amazon India, Nykaa Fashion, or other legitimate retailers.

STRICT RULES:
1. You MUST use Google Search to find the clothing products.
2. Every "link" MUST be a URL actually returned by your search — copy it exactly.
3. If you cannot find a real product, skip it rather than inventing one.
4. Only recommend: Shirts, T-Shirts, Polo Shirts, Hoodies, Sweatshirts, Jackets, Kurtas, 
   Jeans, Chinos, Trousers, Cargo Pants, Shorts, Sneakers, Formal Shoes.
5. Do NOT list accessories as purchasable products — only mention colors in accessory_guidance.

Return ONLY a valid JSON object, no extra text, no markdown:
{{
  "accessory_guidance": {{
    "watch": "recommended color + short reason",
    "shoes": "recommended color + short reason",
    "belt": "recommended color + short reason"
  }},
  "products": [
    {{
      "name": "Exact product name as found",
      "brand": "Brand name",
      "color": "Color of this item",
      "price": "Price as listed (₹)",
      "link": "exact URL from search results"
    }}
  ]
}}

Give 5-6 genuine clothing results, covering different suitable colors.
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(tools=[google_search_tool])
    )

    raw_text = response.text.strip()
    start_idx = raw_text.find('{')
    end_idx = raw_text.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = raw_text[start_idx:end_idx + 1]
    else:
        json_str = raw_text

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"JSON parsing failed. Raw response was:\n{raw_text}")
        result = {"accessory_guidance": {}, "products": []}

    return result

# ============================================
# Routes
# ============================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    photo = request.files["photo"]
    clothing_type = request.form.get("clothing_type", "shirt")
    size = request.form.get("size", "M")
    occasion = request.form.get("occasion", "casual")

    raw_path = os.path.join(UPLOAD_FOLDER, "raw.jpg")
    cropped_path = os.path.join(UPLOAD_FOLDER, "cropped.jpg")
    photo.save(raw_path)

    cropped = crop_face(raw_path, cropped_path)
    if not cropped:
        return jsonify({"error": "No face detected. Try a clearer, well-lit photo."}), 400

    tone, confidence, probabilities = predict_skin_tone(cropped)
    result = get_clothing_suggestions(probabilities, clothing_type, size, occasion)

    return jsonify({
        "skin_tone": tone,
        "confidence": confidence,
        "probabilities": probabilities,
        "accessory_guidance": result.get("accessory_guidance", {}),
        "products": result.get("products", [])
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)