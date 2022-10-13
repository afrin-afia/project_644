import tensorflow as tf
from tensorflow.keras.models import load_model

model= load_model("model_cnn.h5")
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model_cnn.tflite', 'wb') as f:
  f.write(tflite_model)
