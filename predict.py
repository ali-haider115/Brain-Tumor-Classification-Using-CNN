
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model("models/brain_tumor_model.keras")

class_names = sorted(os.listdir("dataset/train"))

import os

image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "p.jpg"
)

print("Image path:", image_path)
print("Image exists:", os.path.isfile(image_path))

if not os.path.isfile(image_path):
    print("Error: Image file not found.")
    exit()

image = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128)
)

image = tf.keras.utils.img_to_array(image)
image = np.expand_dims(image, axis=0)

prediction = model.predict(image, verbose=0)
index = np.argmax(prediction)

print("Predicted Class:", class_names[index])
print("Confidence:", round(float(prediction[0][index]) * 100, 2), "%")
