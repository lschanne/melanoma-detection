from .forms import PatientDataForm
from .constants import (
    SEX_REFUSE,
    AGE_REFUSE,
)

def get_prediction_results(patientDataForm: PatientDataForm) -> dict:
    sex = patientDataForm['sex']
    age = patientDataForm['age']

    if sex == SEX_REFUSE:
        sex = None
    if age == AGE_REFUSE:
        age = None

    return {
        'predicted_probability': 0.1,
    }
