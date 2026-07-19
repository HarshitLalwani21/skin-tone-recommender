import tensorflow as tf
import keras

print("Loading existing model...")
model = keras.models.load_model("skin_tone_model.keras")

print("Converting to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("Saving TFLite model...")
with open("skin_tone_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Done! Check for skin_tone_model.tflite in your folder.")