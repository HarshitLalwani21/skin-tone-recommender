from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, render_template
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN
import cv2
import os
import json
from google import genai
from google.genai import types

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================
# Load Trained Model Once (at startup)
# ============================================
model = keras.models.load_model("skin_tone_model.keras")
class_names = ["deep", "medium", "fair"]  # match training order

# ============================================
# Gemini Client (Web Search via Google Search Grounding)
# ============================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

google_search_tool = types.Tool(google_search=types.GoogleSearch())

detector = MTCNN()

# ============================================
# Face Crop
# ============================================
def crop_face(img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face_crop = img_rgb[y:y+h, x:x+w]
    face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, face_crop_bgr)
    return output_path

# ============================================
# Predict Skin Tone (Using Our Trained CNN Model)
# ============================================
def predict_skin_tone(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.resnet50.preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)[0]

    probabilities = {
        class_names[i]: round(float(prediction[i]) * 100, 2)
        for i in range(len(class_names))
    }

    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]

    return predicted_class, confidence, probabilities

# ============================================
# Get Clothing + Accessory Suggestions
# Gemini decides colors ITSELF from the raw probabilities (no hardcoded table)
# ============================================
def get_clothing_suggestions(probabilities, clothing_type, size, occasion):
    prob_str = ", ".join([f"{k}: {v}%" for k, v in probabilities.items()])

    prompt = f"""
You are an expert men's fashion stylist and color analyst with real-time Google Search access.

A machine learning model analyzed a person's face and produced these skin tone probabilities:
{prob_str}

Based on established color theory and skin undertone matching principles, decide for 
yourself which colors would best complement this specific skin tone profile (treat the 
percentages as a blend, not just the top category — e.g. if it's mostly medium with some 
tan, lean toward colors that suit both).

Then:

1. Search the live web to find REAL, currently listed {clothing_type} products in size {size}, 
   suitable for a {occasion} occasion, in the colors you determined suit this skin tone.

2. Also give brief, minimal accessory color guidance — just one short recommended color each 
   for: watch strap/case, shoes, and belt. Do not search for or list specific accessory 
   products, just the recommended colors with a one-line reason.

Search across multiple online stores — Myntra, Ajio, Flipkart, Amazon India, Nykaa Fashion, 
or any other legitimate retailer. Do not limit yourself to one website.

STRICT RULES:
1. You MUST use Google Search to find the clothing products. Do not answer from memory alone.
2. Every "link" you provide MUST be a URL that was actually returned by your search — 
   copy it exactly as found. Do NOT construct, guess, modify, or shorten any URL.
3. If you cannot find a real product for a given color/type combination after searching, 
   skip it rather than inventing one.
4. Only recommend clothing from: Shirts, T-Shirts, Polo Shirts, Hoodies, Sweatshirts, Jackets, 
   Kurtas, Jeans, Chinos, Trousers, Cargo Pants, Shorts, Sneakers, Formal Shoes.
5. Do NOT search for or list accessories as purchasable products — only mention their colors 
   in the "accessory_guidance" section.

Return ONLY a valid JSON object, with no extra text before or after it, no markdown formatting, 
in this exact format:
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
        config=types.GenerateContentConfig(
            tools=[google_search_tool]
        )
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