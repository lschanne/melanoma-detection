from collections import namedtuple
import os

SEX_REFUSE = 'Refuse to answer'
SEX_MALE = 'Male'
SEX_FEMALE = 'Female'

AGE_REFUSE = ''

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_PATH = os.path.join(STATIC_DIR, 'model.hdf5')

# image size used by keras model
TARGET_IMG_SIZE = (224, 224)

Author = namedtuple('Author', ['name', 'email', 'image'])
AUTHORS = [
    Author('Sharon Cheng', 'hycheng@berkeley.edu', 'images/sharon.jpg'),
    Author('Lingyao Meng', 'lingyaomeng@berkeley.edu', 'images/lingyao.jpg'),
    Author('Luke Schanne', 'lschanne@berkeley.edu', 'images/luke.jpg'),
]

class AnatomicSite:
    choices = [
        'head/neck',
        'upper e4xtremity',
        'lower extremity',
        'torso',
        'palms/soles',
        'oral/genital',
        'not sure',
    ]
