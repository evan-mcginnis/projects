import os
import sys
import logging
import logging.config
import glob

import argparse
import cv2 as cv
import numpy as np

from Factors import Factors, FactorKind, ColorSpace, FactorTypes
from ImageLogger import ImageLogger
from ImageManipulation import ImageManipulation
from ImageProcessing import ImageProcessing
from Performance import Performance
# My Libraries
from VegetationIndex import VegetationIndex
from Classifier import Classifier, HeuristicClassifier
from OptionsFile import OptionsFile
import constants

SECTION_CLASSIFY = "classify"
PARAMETER_FACTOR = "factor"
SAMPLE_SIZE = 20

#from CameraFile import CameraFile

parser = argparse.ArgumentParser("Gila River Classify")

parser.add_argument("-i", "--index", action="store", required=False, default="YCbCrI", choices=VegetationIndex.algorithmChoices, help="Source file or directory")
parser.add_argument("-ini", "--ini", action="store", required=False, default="classify.ini", help="INI file")
parser.add_argument("-f", "--file", action="store", required=True, help="Source file or directory")
parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
parser.add_argument("-o", "--output", action="store", required=False, default=".", help="Output directory")
parser.add_argument("-t", "--threshold", action="store", required=False, default="OTSU", help="Threshold for index")

arguments = parser.parse_args()

# L O G G I N G

if not os.path.isfile(arguments.logging):
    print("Unable to access logging configuration file {}".format(arguments.logging))
    sys.exit(-1)

logging.config.fileConfig(arguments.logging)

log = logging.getLogger(__name__)

# O U T P U T

if not os.path.isdir(arguments.output):
    print(f"Output must be a directory: {arguments.output}")
    sys.exit(-1)

imgLogger = ImageLogger()
imgLogger.connect(arguments.output)

# INI
options = OptionsFile(arguments.ini)
if not options.load():
    print("Unable to load configuration file")
    sys.exit(-1)

if os.path.isdir(arguments.file):
    files = glob.glob(os.path.join(arguments.file, "*.jpg"))
else:
    files = [arguments.file]

if len(files) == 0:
    print("Unable to find any input files")
    sys.exit(-1)

vegetationIndex = VegetationIndex()

# Process each of the files
for file in files:
    image = cv.imread(file, cv.IMREAD_COLOR)
    log.debug(f"Reading image {file}. Size: {image.shape}")
    imgLogger.logImage("original", image)

    #
    # Vegetation Index
    #
    vegetationIndex.SetImage(image)
    vegetationIndex.setup(arguments.index, arguments.threshold)

    log.debug(f"Use threshold: {vegetationIndex.threshold}")
    finalMask = cv.normalize(vegetationIndex.imageMask, None, 0, 255, cv.NORM_MINMAX)
    imgLogger.logImage("mask", finalMask)
    vegetationIndex.applyMask()
    image = vegetationIndex.GetImage()
    finalImage = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    imgLogger.logImage("final", finalImage)

    performance = Performance(arguments.output + "/performance.csv")
    manipulated = ImageManipulation(finalImage, 0, imgLogger)
    manipulated.fontscale = 5
    # log.debug(f"Manipulation object start: {get_size(manipulated)}")
    manipulated.performance = performance
    #
    # Draw a grid on image
    #
    manipulated.drawGrid(SAMPLE_SIZE)
    imgLogger.logImage("grid", manipulated.image)

    manipulated.divideImage(SAMPLE_SIZE)

    # Color space conversions
    manipulated.toGreyscale()
    performance.start()
    manipulated.toYCBCR()
    performance.stopAndRecord(constants.PERF_YCC)
    performance.start()
    manipulated.toHSV()
    performance.stopAndRecord(constants.PERF_HSV)
    performance.start()
    manipulated.toHSI()
    performance.stopAndRecord(constants.PERF_HSI)
    performance.start()
    manipulated.toYIQ()
    performance.stopAndRecord(constants.PERF_YIQ)
    performance.start()
    manipulated.toCIELAB()
    performance.stopAndRecord(constants.PERF_CIELAB)
    manipulated.extractImages()

    # G L C M
    manipulated.progress = True
    manipulated.computeGLCMForImages([constants.NAME_GREYSCALE_IMAGE])
    #manipulated.computeGLCM()

    # P R O C E S S
    processor = ImageProcessing(arguments.output + "/processor.csv")
    allFactors = Factors()
    restrictions = [(FactorTypes.TEXTURE, ColorSpace.GREYSCALE)]
    processor.columns = allFactors.getColumns(["TEXTURE"], ["GLCM"], FactorKind.SCALAR, blacklist=None, restricted=restrictions)
    processor.initialize()
    # Show the progress bar
    processor.progress = True
    # Add the blobs to the processor
    processor.addBlobs(manipulated.blobs)

    # C L A S S I F Y
    factor = options.option(SECTION_CLASSIFY, PARAMETER_FACTOR)
    classify = HeuristicClassifier(processor.df)
    classify.factor = factor
    classify.filename = arguments.output + "/classified.csv"
    classify.classify()
    classify.write()

    # Create overlay map from classifications
    classificationImage = np.zeros_like(image)
    for index, row in classify.df.iterrows():
        components = row['name'].split('-')
        assert len(components) == 3
        imageColumn = int(components[1])
        imageRow = int(components[2])
        if row['type'] == 0:
            classificationImage[imageRow:imageRow + SAMPLE_SIZE ,imageColumn: imageColumn + SAMPLE_SIZE] = (255, 0, 0)
        else:
            classificationImage[imageRow:imageRow + SAMPLE_SIZE ,imageColumn: imageColumn + SAMPLE_SIZE] = (0, 0, 255)

    imgLogger.logImage("classification", classificationImage)

    # H O G
    # manipulated.computeHOG()
