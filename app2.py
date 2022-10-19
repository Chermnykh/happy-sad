import streamlit as st          # lerkvln
from PIL import Image               # lsmb
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf         # lemvlm
import numpy as np              # vkm
# from tensorflow import keras
from tensorflow.keras.models import load_model          # hello
from tensorflow.keras import preprocessing
import time

# from tempfile import NamedTemporaryFile  # lrmve

fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Emotion Classifier')

st.markdown(
    "Welcome to this simple web application that classifies your emotion. Emotions are classified into two classes namely: Happy, Sad.")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "model.h5"
#     IMAGE_SHAPE = (224, 224, 3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_pic = image.resize((150, 150))
    test_pic = tf.keras.preprocessing.image.img_to_array(test_pic)
    test_pic = test_pic/255 # normalization
    test_pic = np.expand_dims(test_pic, axis=0)
    prediction = model.predict(test_pic)
    print(prediction)

    if prediction[0] < 0.5:
        st.markdown("Prediction: Happy")
    else:
        st.markdown("Prediction: Sad")

#     result = f"{class_names[np.argmax(scores)]} with a {(100 * np.max(scores)).round(2)} % confidence."
#     return result


if __name__ == "__main__":
    main()

# #predict
# test_pic = tf.keras.preprocessing.image.load_img(demo_image)
# test_pic = test_pic.resize((150, 150))
# test_pic = tf.keras.preprocessing.image.img_to_array(test_pic)
# test_pic = test_pic/255 # normalization
# test_pic = np.expand_dims(test_pic, axis=0)
# prediction = model.predict(test_pic)
# print(prediction)

# if prediction[0] < 0.5:
#     st.markdown("Prediction: Happy")
# else:
#     st.markdown("Prediction: Sad")

