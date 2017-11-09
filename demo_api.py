import os
import time
import cv2
import numpy as np
import dlib
from predictor import Predictor
from flask import Flask, request, jsonify
from keras.backend.common import image_data_format
 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
# from flask import Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4*1024*1024

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def index():
    return "Hello, World!"
 
@app.route('/api/v1/test', methods=['POST'])
def test():
    file = request.files['file']
    valid_file = file and allowed_file(file.filename)
     
    if not valid_file:
        return jsonify({"ok": False, "message": "file is none or valid"})
     
    image_data = np.fromstring(file.read(), np.uint8)
    result = scan(image_data)
    return jsonify({"ok": True, "data": result})
 
predictor = Predictor()
detector = dlib.get_frontal_face_detector()
def scan(image_data):
    result = []
    
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if img is None:
        return []
    
    input_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = np.shape(input_img_gray)
    
    # detect faces using dlib detector
    detected = detector(input_img_gray, 1)
    faces = np.empty( (len(detected), 64, 64, 3) )
    
    for i, d in enumerate(detected):
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        rectangle = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": w, "h": h}
        result.append({"rectangle": rectangle})
        
        xw1 = max(int(x1 - 0.4 * w), 0)
        yw1 = max(int(y1 - 0.4 * h), 0)
        xw2 = min(int(x2 + 0.4 * w), img_w - 1)
        yw2 = min(int(y2 + 0.4 * h), img_h - 1)
        
        faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (64, 64))
     
    if len(detected) > 0:
        predicted_genders, predicted_ages, predicted_emotion = predictor.inference(faces)
#         print(predicted_genders)
#         print(predicted_ages)
#         print(predicted_emotion)
     
    for i, d in enumerate(detected):
        gender = "female" if predicted_genders[i].argmax() == 0 else "male"
        age = int(predicted_ages[i])
        
        EMOTION_LIST = ["sad", "happy", "surprise", "angry", "neutral"]
        emotion = EMOTION_LIST[predicted_emotion[i].argmax()]
        print(gender+'\t'+str(age)+'\t'+emotion)
        
        attrs = {"age": age, "gender": gender, "emotion": emotion}
        result[i].update(attrs)
    
    # [{'rectangle': {'x1': 10, 'y1': 24, 'x2': 140, 'y2': 154, 'w': 130, 'h': 130}, 'age': 21, 'gender': 'female', 'emotion': 'neutral'}]  
    return result
 
if __name__ =='__main__':
    app.run(host = '0.0.0.0', port = 9527)