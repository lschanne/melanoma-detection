from wtforms import (
    FileField,
    Form,
    IntegerField,
    RadioField,
    SubmitField, 
    validators,
)

from .constants import (
    AnatomicSite,
    SEX_FEMALE,
    SEX_MALE,
    SEX_REFUSE,
)

class PatientData(Form):
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
            validators.NumberRange(
                min=0, message='Age must be greater than 0.',
            ),
            validators.Optional(),
        ]
    )

    image_file = FileField(
        '[Required] Image file of mole the mole:',
        validators=[
            # TODO
            # validators.regexp(
            #     r'^.+(jpe?g|png|gif|bmp)$',
            #     message='An image file is required (jpg, png, gif, bmp).'
            # ),
        ],
    )

    anatomic_site = RadioField(
        'Select the anatomic site of the mole/lesion:',
        choices=AnatomicSite.choices,
    )

    submit = SubmitField("Enter")
