#
# C L A S S I F I E R
#
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._filename = "classified.csv"

    @abstractmethod
    def classify(self):
        pass

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame):
        self._df = df

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, targetFilename: str):
        self._filename = targetFilename

    def write(self):
        self._df.to_csv(self._filename, encoding="UTF-8", index=False )

class HeuristicClassifier(Classifier):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self._factor = ""

    @property
    def factor(self) -> str:
        return self._factor

    @factor.setter
    def factor(self, targetFactor: str):
        self._factor = targetFactor


    def classify(self):
        """
        Classify by the supplied factor. The factor's values are assumed to be a float.
        :param factor: The name of the factor to use
        """

        # Sanity check
        assert self._factor in self._df.columns

        self._df[self._factor] = self._df[self._factor].astype(float)
        # Use ASM values to classify into two classes
        midpoint = (self._df[self._factor].max() - self._df[self._factor].min()) / 2.0
        self._df['type'] = np.where(self._df[self._factor] > midpoint, 0, 1)
        #self._df.where(self._df['self._factor'] > midpoint, other = 1, inplace=True)

