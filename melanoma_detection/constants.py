from collections import namedtuple

SEX_REFUSE = 'Refuse to answer'
SEX_MALE = 'Male'
SEX_FEMALE = 'Female'

AGE_REFUSE = ''

Author = namedtuple('Author', ['name', 'email', 'image'])
AUTHORS = [
    Author('Sharon Cheng', 'hycheng@berkeley.edu', 'images/sharon.jpg'),
]
