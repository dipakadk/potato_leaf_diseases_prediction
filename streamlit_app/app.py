import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your pre-trained image classification model
model = tf.keras.models.load_model('potato_models.h5')

# Define class labels (replace with your actual class labels)
class_labels = ["Early_blight", "Late_blight", "Healthy"]

def main():
    st.title("Potato Leaf Diseases Prediction System")

    # Upload image through Streamlit widget
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for model prediction
        image = np.array(image)
        image = tf.image.resize(image, (256, 256))  # Resize to match your model's input size
        image = image / 255.0  # Normalize

        # Make predictions
        predictions = model.predict(np.expand_dims(image, axis=0))

        # Display class probabilities
        st.write("Class Probabilities:")
        for i, prob in enumerate(predictions[0]):
            st.write(f"{class_labels[i]}: {prob:.2f}")

        # Get the predicted class
        predicted_class = np.argmax(predictions)
        st.write(f"Predicted Class: {class_labels[predicted_class]}")

if __name__ == "__main__":
    main()
