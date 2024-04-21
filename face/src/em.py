import onnx
import onnxruntime
import cv2
import numpy as np
from src.det import FaceDet

class Emotional(FaceDet):
    def __init__(self):
        self.facedet = FaceDet()
        self.session = onnxruntime.InferenceSession('models/emotional.onnx',providers=['CPUExecutionProvider'])
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.mean=[0.5, 0.5, 0.5]
        self.std=[0.5, 0.5, 0.5]
        self.img_size = 112
        self.idx_to_class = {
            0: 'Гнев', 
            1: 'Презрение', 
            2: 'Отвращение', 
            3: 'Страх', 
            4: 'Счастье', 
            5: 'Нейтральность', 
            6: 'Грусть', 
            7: 'Удивление'
        }
        
    def __call__(self, img):
        face_im = self.facedet(img)
        face_im=cv2.resize(face_im,(self.img_size,self.img_size))/255
        for i in range(3):
            face_im[..., i] = (face_im[..., i]-self.mean[i])/self.std[i]
        face_im = face_im.transpose(2, 0, 1).astype("float32")[np.newaxis,...]
        outputs = self.session.run(self.output_names, {self.input_names[0]:face_im})
        return self.idx_to_class[np.argmax(outputs)]


