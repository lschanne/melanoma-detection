from typing import Union

from .forms import PatientData
from .constants import (
    SEX_REFUSE,
    AGE_REFUSE,
)

def get_prediction_results(
    sex: Union[str, None],
    age: Union[int, None],
    image_data: bytes,
) -> dict:
    '''
    Sends inputs to the model and return relevant results.

    Parameters
    ----------
    sex: str
        sex of patient; 'Male', 'Female', or None
    age: int
        age of the patient; can be None
    image_data: bytes
        binary contents from file with image of mole
    
    Returns
    -------
    dict:
        predicted_probability: float
            the predicted probability that the mole is malignant
    '''
    # TODO what metrics should we report?
    return {
        'predicted_probability': 0.1,
    }

def process_form(request: 'flask.request', form: PatientData) -> dict:
    '''
    Extract model inputs from the PatientData and the incoming http request.

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
        image_data: bytes
            binary contents from file with image of mole
    '''
    sex = request.form.get('sex', SEX_REFUSE)
    age = request.form['age']
    image_data = request.files[form.image_file.name].read()

    if sex == SEX_REFUSE:
        sex = None
    if age == AGE_REFUSE:
        age = None

    return {
        'sex': sex,
        'age': age,
        'image_data': image_data,
    }