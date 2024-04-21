import onnx
import onnxruntime
import cv2
import numpy as np

class FaceDet():
    def __init__(self):
        self.session = onnxruntime.InferenceSession('models/faceDet.onnx',providers=['CPUExecutionProvider'])
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self. model_inputs))]
        self.model_outputs = self.session.get_outputs()
        self.output_names = [self.model_outputs[i].name for i in range(len(self. model_outputs))]
    def __call__(self,orig_image):
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        outputs = self.session.run(self.output_names, {self.input_names[0]:image})
        scores = outputs[0]
        boxes = outputs[1]
        confidence_threshold = 0.8
        indices = cv2.dnn.NMSBoxes(boxes[0], scores[0][:, 1],confidence_threshold, 0.4) 
        for i in indices:
            box = boxes[0][i]
            x1, y1, x2, y2 = box[:4] * np.array([orig_image.shape[1],orig_image.shape[0], orig_image.shape[1], orig_image.shape[0]])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_im = orig_image[y1:y2, x1:x2] 
            return face_im