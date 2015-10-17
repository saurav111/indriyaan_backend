#
#    See <http://www.opensource.org/licenses/bsd-license>
import logging
# cv2 and helper:
import cv2
import os
from helper.common import *
from helper.video import *
# add facerec to system path
import sys
sys.path.append("../..")
# facerec imports
from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.validation import KFoldCrossValidation
from facerec.serialization import save_model, load_model
import numpy as np

# for face detection (you can also use OpenCV2 directly):
# from facedet.detector import CascadedDetector


def detailsOfRecognizedPeople(image):
    FACE_DETECTOR_PATH = "{base_path}/haarcascade_frontalface_alt2.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    MODEL_PATH = "{base_path}/robin_saurav_final.pkl".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    class App(object):
        def __init__(self, model, cascade_filename):
            self.model = model
            self.cascade_filename = cascade_filename

        def run(self,frame):
            # Resize the frame to half the original size for speeding up the detection process:
            img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
            imgout = img.copy()
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            faceCascade = cv2.CascadeClassifier(self.cascade_filename)
            rects = faceCascade.detectMultiScale(img,minNeighbors=5, scaleFactor=1.2,minSize=(30,30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
            rects = np.array(rects)

            data_frame = {}
            person_names = []
            for i,r in enumerate(rects):
                x,y,w,h = r
                # (1) Get face, (2) Convert to grayscale & (3) resize to image_size:
                face = img[y:y+h, x:x+w]
                # face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, self.model.image_size, interpolation = cv2.INTER_CUBIC)
                # Get a prediction from the model:
                prediction = self.model.predict(face)[0]
                person_names.append(self.model.subject_names[prediction])

            data_frame.update({"num_faces":len(rects),"faces":rects.tolist(),"identities":person_names,"success":True})
            return data_frame

    model_filename = MODEL_PATH
    print "Loading the model..."
    model = load_model(model_filename)
    recognizer = App(model=model,cascade_filename=FACE_DETECTOR_PATH)
    return recognizer.run(image)