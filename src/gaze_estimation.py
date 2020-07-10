import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IECore
import math
from model import Model

'''
This is the class for the Gaze Estimation Model.
contains mefthods for loading, checking, and running inference and 
methods to pre-process the inputs and detections to the model
'''

class GazeEstimation(Model):
    '''
    Class for the Gaze Estimation Model.
    '''    
    def predict(self, left_eye_image, right_eye_image, hpa_cords,image):
        '''
        does predictions from the input.
        '''
        try:
            left_eye_image  = self.preprocess_input(left_eye_image)
            right_eye_image = self.preprocess_input(right_eye_image)
            self.exec_network.start_async(0, inputs={'left_eye_image': left_eye_image,'right_eye_image': right_eye_image,'head_pose_angles': hpa_cords})
            if self.exec_network.requests[0].wait(-1) == 0:
                outputs = self.exec_network.requests[0].outputs
                mouse_coord, gaze_vector = self.preprocess_output(outputs, hpa_cords,image)
        except Exception as e:
            log.error('Gaze Estimation Model| Error during the prediction '+ str(e))
        return mouse_coord, gaze_vector
    
    
    def preprocess_input(self,image):
        ''' 
        Allow us to pre-process the inputs before feeding them into the model.
        '''
        try:
            self.image = cv2.resize(image, (60,60))
            self.image = self.image.transpose((2,0,1))
            self.image = self.image.reshape(1, *self.image.shape)
            return self.image
        except Exception as e:
            log.error("Gaze Estimation | Error on preprocessing  of input " + str(e))
        
            

    def preprocess_output(self, outputs,hpa ,image):
        '''
        The detections of the model need to be preprocessed before they can 
        be used by other models or be used to change the mouse pointer.
        we can do that in here 
        '''
        gaze_vector = outputs[self.output_blob][0]
        mouse_coord = (0, 0)
        try:
            angle_r_fc = hpa[2]
            sin_r = math.sin(angle_r_fc * math.pi / 180.0)
            cos_r = math.cos(angle_r_fc * math.pi / 180.0)
            x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
            y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
            mouse_coord = (x, y)
            if 'gaze_estimation' == self.flags:
                cv2.putText(image,"Gaze Vector : x= {:.2f} , y= {:.2f} , z= {:.2f}".format(gaze_vector[0], gaze_vector[1], gaze_vector[2]),(15, 100),cv2.FONT_HERSHEY_COMPLEX,1, (255, 255, 255), 3)
        except Exception as e:
            log.error("Gaze Estimation Model | Error during the preprocessing of output" + str(e))
        return mouse_coord, gaze_vector