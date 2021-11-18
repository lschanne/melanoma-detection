'''
Custom classes for data preprocessing.
'''

import numpy as np
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
        for col in range(X.shape[1]):
            encoder = LabelEncoder()
            encoder.fit(X[:,col])
            self._encoders[col] = encoder
        return self
    
    def transform(self, X):
        new_columns = []
        for col, encoder in self._encoders.items():
            new_columns.append(encoder.transform(X[:,col]))
        return np.array(new_columns).T
        