from wtforms import (
    Form,
    RadioField,
    validators,
    SubmitField, 
    IntegerField,
)

from .constants import (
    SEX_REFUSE,
    SEX_MALE,
    SEX_FEMALE,
)

class PatientDataForm(Form):
    '''
    Form for collecting the image of the mole and other patient data.
    '''
    sex = RadioField(
        '[Optional] Select the sex of the patient:',
        # TODO: should we have more choices? I doubt the data will have more
        # representation, so maybe we should just have a disclaimer
        choices=[SEX_REFUSE, SEX_MALE, SEX_FEMALE],
        validators=[validators.Optional()],
    )

    age = IntegerField(
        '[Optional] Enter the age of the patient:',
        validators=[
            validators.NumberRange(min=0, message='Age must be greater than 0'),
            validators.Optional(),
        ]
    )

    submit = SubmitField("Enter")
