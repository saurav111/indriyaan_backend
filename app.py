from flask import Flask,request,Response,json
from flask.ext.cors import CORS
import face_recognition
import cv2

app = Flask(__name__)
CORS(app)
@app.route("/", methods=["POST"])
def hello():
    imageData = request.form['image']
    imageData = imageData[23:]
    fh = open('sth.jpeg','wb')
    fh.write(imageData.decode('base64'))
    fh.close()
    imageData = cv2.imread('sth.jpeg')
    jsonData = face_recognition.detailsOfRecognizedPeople(imageData)

    js = json.dumps(jsonData)
    resp = Response(js, status=200, mimetype='application/json')
    print resp
    return resp

if __name__ == "__main__":
    app.run(debug=True)
