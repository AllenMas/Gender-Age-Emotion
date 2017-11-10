import os
import shutil
import time
import cv2
import numpy as np
from predictor import Predictor
from flask import Flask, request, jsonify
import random
import uuid
 
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
    start = time.time()
    result = scan(image_data)
    end = time.time()
    print("GAE server time: %f"%(end-start))
    return jsonify({"ok": True, "data": result})
 
predictor = Predictor()
def scan(image_data):
    result = []
    
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if img is None:
        return []
    
    face = np.empty((1, 64, 64, 3))
    face[0, :, :, :] = cv2.resize(img, (64, 64))
    
    # upload image to AWS s3
    if(probility(5)==True):
        if(os.path.exists("tmp")==True):
            shutil.rmtree("tmp")
        os.mkdir("tmp")
        name = get_unique_filename()+".jpg"
        cv2.imwrite(os.path.join("tmp", name), img)
        os.system("aws s3 cp "+"tmp/%s "%(name)+ "s3://ssd-robby/Wuxiang/GAE_Imgs/")
            
    # prediction
    start = time.time()
    predicted_genders, predicted_ages, predicted_emotion = predictor.inference(face)
    end = time.time()
    
    # result
    gender = "female" if predicted_genders[0].argmax() == 0 else "male"
    age = int(predicted_ages[0])
    
    EMOTION_LIST = ["sad", "happy", "surprise", "angry", "neutral"]
    emotion = EMOTION_LIST[predicted_emotion[0].argmax()]
    print(gender+'\t'+str(age)+'\t'+emotion+'\t'+str(end-start))
    
    attrs = {"rectangle": {"x1": 0, "y1": 0, "x2": 63, "y2": 63, "w": 64, "h": 64},"age": age, "gender": gender, "emotion": emotion}
    result.append(attrs)
     
    return result

def probility(i):
    k = random.randint(1, i)
    if k==1:
        return True
    else:
        return False

def get_unique_filename():
    timestr = time.strftime("%Y%m%d%H%M%S")
    return timestr + str(uuid.uuid4()).replace('-','')
    
if __name__ =='__main__':
    app.run(host = '0.0.0.0', port = 9527)