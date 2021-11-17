from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D

from .constants import AnatomicSite, MODEL_PATH, PREPROCESSOR_PATH
from .forms import PatientData

class Model:
    TARGET_IMG_SIZE = (224, 224)
    RESNET_SHAPE = (*TARGET_IMG_SIZE, 3)
    
    def __init__(self):
        with open(PREPROCESSOR_PATH, 'rb') as f:
            self._preprocessor = pickle.load(f)
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

        if sex == SEX_REFUSE:
            sex = np.nan
        if age == AGE_REFUSE:
            age = np.nan
        if anatomic_site == AnatomicSite.REFUSE:
            anatomic_site = np.nan

        img = self._preprocess_image(image_file.read())
        image_features = model_ResNet50.predict(img)
        full_features = pd.concat(
            pd.DataFrame(
                {
                    'sex': sex,
                    'age_approx': age,
                    'anatom_site_general_challenge': anatomic_site,
                },
                index=0
            ),
            pd.DataFrame(image_features),
            axis=1,
        )
        transformed_features = self._preprocessor.transform(full_features)
        prediction = self.model.predict(
            pd.DataFrame(transformed_features)
        )[0, 1]
        return {
            'sex': sex,
            'age': age,
            'file_name': image_file.filename,
            'anatomic_site': anatomic_site,
            'predicted_probability': prediction,
        }

    @classmethod
    def _preprocess_image(cls, image_data: bytes) -> np.ndarray:
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
        img = img.resize(cls.TARGET_IMG_SIZE, Image.NEAREST)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = ImageDataGenerator(rescale=1./255).flow(img, batch_size=1)[0]
        return img


### START: Custom classes used by the preprocessor ###
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._encoders = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self._encoders[col] = encoder
        return self
    
    def transform(self, X):
        X_new = pd.DataFrame()
        for col, encoder in self._encoders.items():
            X_new[col] = encoder.transform(X[col])
        return X_new
### END: Custom classes used by the preprocessor ###
