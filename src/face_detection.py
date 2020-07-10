import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
from model import Model

'''
This is the class for a FaceDetection Model.
contains mefthods for loading, checking, and running inference and 
methods to pre-process the inputs and detections to the model
'''

class FaceDetection(Model):
    '''
    Class for the Face Detection Model.
    '''
        
    def predict(self, image):
        '''
        does predictions from the input.
        '''
        try:
            img_processed = self.preprocess_input(image)
            results = self.exec_network.infer({self.input_blob:img_processed})
            detections = results[self.output_blob]
            self.points, self.image = self.preprocess_output(detections,image)
            try:
                cropped_face = image[self.points[0][1]:self.points[0][3], self.points[0][0]:self.points[0][2]]
            except Exception as e:
                log.error('Could not detect right eye in the frame, Index is out of range')
                exit()
            return image,cropped_face,detections
        except Exception as e:
            log.error('Face detection failed, no face detected in the frame')
            exit()

    def preprocess_input(self, image):
        ''' 
        Allow us to pre-process the inputs before feeding them into the model.
        '''
        self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        self.image = self.image.transpose((2,0,1))
        self.image = self.image.reshape(1, *self.image.shape)
        return self.image

    def preprocess_output(self, detections,image):
        '''
        The detections of the model need to be preprocessed before they can 
        be used by other models or be used to change the mouse pointer.
        we can do that in here 
        '''
        points=[]
        for box in detections[0][0]: 
            conf = box[2]
            if conf > self.prob_threshold :
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                points.append((xmin, ymin, xmax, ymax))
                if 'face_detection'  == self.flags :
                    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)       
        return points, image
