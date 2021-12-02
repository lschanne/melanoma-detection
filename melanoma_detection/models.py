from io import BytesIO
import joblib
from keras.models import load_model
from keras.preprocessing.image import img_to_array, ImageDataGenerator
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D

from .constants import (
    AGE_REFUSE, AnatomicSite, MODEL_PATH, PREPROCESSOR_PATH, SEX_REFUSE, USE_LR
)
from .forms import PatientData

class Model:
    TARGET_IMG_SIZE = (224, 224)
    RESNET_SHAPE = (*TARGET_IMG_SIZE, 3)
    
    def __init__(self):
        with open(PREPROCESSOR_PATH, 'rb') as f:
            self._preprocessor = pickle.load(f)
        if USE_LR:
            self._model = joblib.load(MODEL_PATH)
        else:
            self._model = load_model(MODEL_PATH)
        self._resnet50 = Sequential([
            ResNet50(
                input_shape=self.RESNET_SHAPE,
                weights='imagenet',
                include_top=False,
            ),
            GlobalAveragePooling2D()
        ])

    def predict(self, request: 'flask.request', form: PatientData):
        '''
        Extract model inputs from input form and make a prediction.

        Parameters
        ----------
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
        anatomic_site = request.form.get('anatomic_site', AnatomicSite.REFUSE)
        image_file = request.files[form.image_file.name]

        str_fill_value = 'unknown'
        counterfactuals = {}
        if sex == SEX_REFUSE:
            sex = str_fill_value
            counterfactuals['sex'] = ['male', 'female']
        else:
            sex = sex.lower()

        if age == AGE_REFUSE:
            age = np.nan
            counterfactuals['age'] = [20, 30, 40, 50, 60, 70]
        if anatomic_site == AnatomicSite.REFUSE:
            anatomic_site = str_fill_value
            counterfactuals['anatomic site'] = AnatomicSite.sites
        else:
            anatomic_site = anatomic_site.lower()
        


        img = self._preprocess_image(image_file.read())
        image_features = self._resnet50.predict(img)

        key_map = {
            'age': 'age_approx',
            'anatomic site': 'anatomic_site_general_challenge',
        }
        metadata = {
            'sex': sex,
            'age_approx': age,
            'anatom_site_general_challenge': anatomic_site,
        }
        rows = [metadata]
        for key, values in counterfactuals.items():
            new_key = key_map.get(key, key)
            for value in values:
                new_row = dict(metadata)
                new_row[new_key] = value
                rows.append(new_row)

        full_features = pd.concat(
            (
                pd.DataFrame(rows),
                pd.DataFrame(
                    np.repeat(image_features, len(rows), axis=0),
                    columns=map(str, range(len(image_features[0]))),
                ),
            ),
            axis=1,
        )
        transformed_features = self._preprocessor.transform(full_features)

        if USE_LR:
            predictions = self._model.predict_proba(
                pd.DataFrame(transformed_features)
            )
        else:
            predictions = self._model.predict(
                pd.DataFrame(transformed_features)
            )
        
        predictions = [
            f'{100 * prediction:.2f}%' for prediction in predictions.flatten()
        ]
        counterfactual_predictions = []
        iii = 0
        for key, values in counterfactuals.items():
            for value in values:
                iii += 1
                counterfactual_predictions.append(
                    (key, value, predictions[iii])
                )

        return {
            'sex': '' if sex is np.nan else sex,
            'age': '' if age is np.nan else age,
            'file_name': image_file.filename,
            'anatomic_site': anatomic_site,
            'predicted_probability': predictions[0],
            'counterfactuals': counterfactual_predictions,
        }

    @classmethod
    def _preprocess_image(cls, image_data: bytes) -> np.ndarray:
        '''
        Convert the binary contents of the image file into a matrix for the
        model to use.
        
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
        with Image.open(BytesIO(image_data)) as f:
            img = f.convert('RGB')
            img = img.resize(cls.TARGET_IMG_SIZE, Image.NEAREST)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = ImageDataGenerator(rescale=1./255).flow(img, batch_size=1)[0]
            return img
