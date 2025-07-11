import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

# Load model once
model = load_model('waste_model.h5')
class_names = ['metal', 'organic', 'plastic']

def predict_waste(file_storage):
    # Read image as byte stream
    img_bytes = file_storage.read()
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])

    return class_names[class_idx]
