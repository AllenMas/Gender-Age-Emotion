import os
import time
import cv2
import numpy as np
import dlib
from predictor import Predictor

def face_detector(image, img_size):
    img = cv2.imread(image)
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)
      
    detector = dlib.get_frontal_face_detector()
    detected = detector(input_img, 1)
      
    face = np.empty( (1, img_size, img_size, 3) )
    if len(detected) > 0:
          
          
        x1, y1, x2, y2, w, h = detected[0].left(), detected[0].top(), detected[0].right() + 1, detected[0].bottom() + 1, detected[0].width(), detected[0].height()
  
        xw1 = max(int(x1 - 0.4 * w), 0)
        yw1 = max(int(y1 - 0.4 * h), 0)
        xw2 = min(int(x2 + 0.4 * w), img_w - 1)
        yw2 = min(int(y2 + 0.4 * h), img_h - 1)
          
        face[0,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
          
        return face, x1, y1, x2, y2, True
      
    return face, None, None, None, None, False
  
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
      
predictor = Predictor()
imgs = os.listdir("testsample")
for img in imgs:
    face, x1, y1, x2, y2, condition = face_detector(os.path.join("testsample", img), 64)
    if condition == True:
        start = time.clock()
        predicted_genders, predicted_ages, predicted_emotion = predictor.inference(face)
        end = time.clock()
        
        gender = "female" if predicted_genders[0].argmax() == 0 else "male"
        age = int(predicted_ages[0])
        EMOTION_LIST = ["sad", "happy", "surprise", "angry", "neutral"]
        emotion = EMOTION_LIST[predicted_emotion[0].argmax()]
        
        print("time: %f"%(end-start)+'\t'+img+'\t'+gender+'\t'+str(age)+'\t'+emotion)
          
        im = cv2.imread(os.path.join("testsample", img))
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
        draw_label(im, (x1, y2), gender+', '+str(age)+', '+emotion)
        cv2.imwrite(os.path.join("result", img), im)