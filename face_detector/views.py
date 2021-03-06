# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
import face_recognition
import logging
# cv2 and helper:
import cv2
from helper.common import *
from helper.video import *
try:
    import json
except ImportError:
    import simplejson as json


# add facerec to system path
# import sys
# sys.path.append("../..")
# facerec imports
# for face detection (you can also use OpenCV2 directly):
# from facedet.detector import CascadedDetector

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/haarcascade_frontalface_alt2.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
	
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])

		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)

			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)

			# load the image and convert
			image = _grab_image(url=url)

		# convert the image to grayscale, load the face cascade detector,
		# and detect faces in the image
		# data = detailsOfRecognizedImage(image)
		data = face_recognition.detailsOfRecognizedPeople(image)
				
	# return a JSON response
	return JsonResponse(data)

def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)

	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()

		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()

		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image