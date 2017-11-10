import numpy as np
from wide_resnet import WideResNet
from emotion_net import *
from gender_age_net import *
from keras.models import Model

class Predictor:
    def __init__(self):
        self.img_size = 64
        self.model = WideResNet(self.img_size, depth=16, k=8)()
        self.model.load_weights("models/WRN_16_8.h5")
        
        self.front_model = Model(inputs = self.model.input, outputs = self.model.get_layer('flatten_1').output)
        
        # back_model predict gender and age
        self.model_ga = gender_age_net()
        self.model_ga.load_weights("models/WRN_16_8.h5", by_name=True)
        
        # back_model predict emotion
        self.model_e = emotion_net()
        self.model_e.load_weights("back_models/WRN_16_8.h5")
    
    def inference(self, face):
#         AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
#         EMOTION_LIST = ["sad", "happy", "surprise", "angry", "neutral"]        
        flatten_1 = self.front_model.predict(face)
        
        result0 = self.model_ga.predict(flatten_1)
        predicted_genders = result0[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = result0[1].dot(ages).flatten()
        
        result1 = self.model_e.predict(flatten_1)
        predicted_emotion = result1
        
        return predicted_genders, predicted_ages, predicted_emotion
#         return "F" if predicted_genders[0].argmax()==0 else "M", str(int(predicted_ages)), EMOTION_LIST[predicted_emotion[0].argmax()]