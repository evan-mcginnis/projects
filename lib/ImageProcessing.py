#
# I M A G E  P R O C E S S I N G
#
from typing import Dict, List
import os.path
import re
import matplotlib.pyplot as plt
import constants
import logging
import numpy as np
import pandas as pd
from rich.progress import Progress

class ImageProcessing:
    def __init__(self, filename: str):
        """
        Various image processing functions
        :param filename: base filename for results
        """
        self.log = logging.getLogger(__name__)
        self._filename = filename
        self._resultsFilename = self._filename + constants.EXTENSION_CSV
        self._normalizedFilename = self._filename + constants.DELIMETER + constants.FILENAME_NORMALIZED + constants.EXTENSION_CSV
        self._dataframeFilename = self._filename + constants.DELIMETER + constants.FILENAME_DATAFRAME + constants.EXTENSION_CSV
        self._dataframeFilenameVectors = self._filename + constants.DELIMETER + constants.FILENAME_DATAFRAME + constants.DELIMETER + constants.VECTOR + constants.EXTENSION_CSV

        self._progress = False
        # The column names to use
        self._columns = [constants.NAME_NAME,
                         constants.NAME_NUMBER,
                         constants.NAME_TYPE]


        # Columns to exclude from translations like normalizing values

        self._exclude = [constants.NAME_NAME, constants.NAME_NUMBER, constants.NAME_TYPE]
        self._blobDF = pd.DataFrame(columns=self._columns)

    @property
    def columns(self) -> List[str]:
        return self._columns

    @columns.setter
    def columns(self, columns: List[str]):
        self._columns.extend(columns)
        self._blobDF = pd.DataFrame(columns=self._columns)

    @property
    def progress(self) -> bool:
        """
        Long running operations use a progress bar.
        :return:
        """
        return self._progress

    @progress.setter
    def progress(self, showProgress: bool):
        """
        Indicate if long running operations show use a progress bar.
        :param showProgress:
        """
        self._progress = showProgress

    @property
    def df(self) -> pd.DataFrame:
        return self._blobDF

    def initialize(self) -> (bool, str):
        """
        Initialize reporting by conforming access to and truncating files
        :return: (bool, str)
        """
        for filename in [self._resultsFilename, self._normalizedFilename, self._dataframeFilename]:
            if os.path.isfile(filename):
                return False, f"Results file {filename} exists. Will not overwrite."

        return True, "OK"

    def reset(self):
        """
        Resets the object for another processing operation.
        """
        self._blobDF = pd.DataFrame(columns=self._columns)


    def addBlobs(self, blobs: Dict[str, float]):
        """
        Add the blob dictionary to the global list of everything seen so far.
        :param ctx: Context for image
        :param sequence: The sequence number of this image
        :param blobs: The detected and classified items in the image
        """
        progress = Progress()
        if self._progress:
            progress.start()

        task1 = progress.add_task("Add blobs", total=len(blobs))

        for blobName, blobAttributes in blobs.items():
            progress.update(task1, advance=1.0)
            attributes = {}
            attributes[constants.NAME_NAME] = blobName
            attributes[constants.NAME_NUMBER] = blobName.split(constants.NAME_BLOB)[1]
            attributes[constants.NAME_TYPE] = 0
            # Add only the columns needed
            for attributeName, attributeValue in blobAttributes.items():
                # Attribute names will be in form <color space>_<attribute>_<angle>
                components = attributeName.split(constants.DELIMETER)
                # Scalars
                self.log.debug(f"Attribute: {attributeName} {components}")
                if attributeName in self._columns and attributeName not in self._exclude:
                    attributes[attributeName] = attributeValue
            # Scalars
            self._blobDF = self._blobDF.append(attributes, ignore_index=True)

        if self._progress:
            progress.stop()

        # write out the data, appending to the file if it is already there
        # Previous versions just kept everything in a dataframe and wrote it out in the end,
        # but that caused problems for large data sets
        if os.path.isfile(self._dataframeFilename):
            self._blobDF.to_csv(self._resultsFilename, mode='a', header=False, encoding="UTF-8", index=False)
        else:
            self._blobDF.to_csv(self._resultsFilename, encoding="UTF-8", index=False)




    def _normalize(self):
        """
        Normalize the data
        """
        # apply normalization techniques
        for column in self._columns:
            if column not in self._exclude:
                try:
                    minimum = self._blobDF[column].min()
                    maximum = self._blobDF[column].max()
                    self.log.debug(f"Checking: {column}")
                    if type(minimum) == pd.Series:
                        if minimum[0] == maximum[0]:
                            self.log.error(f"Normalize (Series): Min == Max for {column}")
                            self._blobDF[column] = 1
                    elif minimum == maximum:
                        self.log.error(f"Normalize: Min == Max for {column}")
                        self._blobDF[column] = 1
                    else:
                        self._blobDF[column] = (self._blobDF[column] - self._blobDF[column].min()) / (
                                    self._blobDF[column].max() - self._blobDF[column].min())
                except ZeroDivisionError:
                    self.log.error("Division by zero error for column {}".format(column))
                    self._blobDF[column] = 1
        return
