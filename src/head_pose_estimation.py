import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IECore
from model import Model

'''
This is the class for a Head Pose Estimation Model.
contains mefthods for loading, checking, and running inference and 
methods to pre-process the inputs and detections to the model
'''
class HeadPoseEstimation(Model):
    '''
    Class for the HeadPoseEstimation Model.
    '''
        
    def predict(self, face_crop,frame):
        '''
        does predictions from the input.
        '''
        try:
            img_processed = self.preprocess_input(face_crop)
            #tart an asynchronous request ###
            self.exec_network.start_async(0, inputs={self.input_blob: img_processed})
            if self.exec_network.requests[0].wait(-1) == 0:
                outputs = self.exec_network.requests[0].outputs
                results = self.preprocess_output(outputs,frame)
        except Exception as e:
            log.error('Head Pose Estimation | No prediction found responding to the input')
            exit()
        return results
    
    def preprocess_input(self, image):
        ''' 
        Allow us to pre-process the inputs before feeding them into the model.
        '''
        try:
            self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            self.image = self.image.transpose((2,0,1))
            self.image = self.image.reshape(1, *self.image.shape)
        except Exception as e:
            log.ERROR("Error on preprocessing the model " + str(self.model_xml) + str(e))
        return self.image
            

    def preprocess_output(self, outputs,frame):
        '''
        The detections of the model need to be preprocessed before they can 
        be used by other models or be used to change the mouse pointer.
        we can do that in here 
        '''
        try:
            results = []
            results.append(outputs['angle_r_fc'][0][0])
            results.append(outputs['angle_p_fc'][0][0])
            results.append(outputs['angle_y_fc'][0][0])
            if 'pose_estimation' == self.flags :
                cv2.putText(frame,"Pose Angles: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(results[0],  results[1],  results[2]),(15, 150),cv2.FONT_HERSHEY_COMPLEX,1, (0, 255, 0), 3)
        except Exception as e:
            log.error("Head Pose Estimation Model| Error when processing output" + str(e))
        return results