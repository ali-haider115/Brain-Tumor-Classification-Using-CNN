
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)

# Load trained model
model = tf.keras.models.load_model(
    "models/brain_tumor_model.keras"
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("Test Loss:", round(loss, 4))

# Predict test images
y_true = np.concatenate([
    labels.numpy() for images, labels in test_data
])

y_pred = np.argmax(model.predict(test_data), axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=test_data.class_names
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
