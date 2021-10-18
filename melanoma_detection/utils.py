from io import BytesIO
from typing import Union

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras import Model
import numpy as np
from PIL import Image

from .forms import PatientData
from .constants import (
    AGE_REFUSE,
    MODEL_PATH,
    SEX_REFUSE,
    TARGET_IMG_SIZE,
)

def get_prediction_results(
    model: Model, request: 'flask.request', form: PatientData,
) -> dict:
    '''
    Extract model inputs from input form and process the resulting prediction.

    Parameters
    ----------
    model: Model
        the pre-trained keras model for making predictions
    request:
        incoming POST request
    form:
        form asking for model inputs
    
    Returns
    -------
    dict
        sex: str
            sex of patient; 'Male', 'Female', or None
        age: int
            age of the patient; can be None
        file_name: str
            name of the uploaded image file
        image_data: bytes
            binary contents from file with image of mole
        anatomic_site: str
            the anatomic site at which the image was taken
        predicted_probability: float
            the predicted probability that the mole is malignant
    '''
    sex = request.form.get('sex', SEX_REFUSE)
    age = request.form['age']

    # if sex == SEX_REFUSE:
    #     sex = None
    # if age == AGE_REFUSE:
    #     age = None

    img = preprocess_image(request.files[form.image_file.name].read())
    return {
        'sex': sex,
        'age': age,
        'file_name': request.files[form.image_file.name].filename,
        'anatomic_site': request.form['anatomic_site'],
        'predicted_probability': model.predict(img)[0, 1],
    }

def preprocess_image(image_data: bytes) -> np.ndarray:
    '''
    Convert the binary contents of the image file into a matrix for the model
    to use.
    
    https://github.com/keras-team/keras/issues/11684

    Parameters
    ----------
    image_data: bytes
        the binary contents of the image file
    
    Returns
    -------
    np.ndarray
        matrix form of image
    '''
    img = Image.open(BytesIO(image_data))
    img = img.convert('RGB')
    img = img.resize(TARGET_IMG_SIZE, Image.NEAREST)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def load_keras_model() -> Model:
    '''
    Load in the pre-trained model

    Returns
    -------
    Model
        the pre-trained keras model for making predictions
    '''
    return load_model(MODEL_PATH)
