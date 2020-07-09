import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IECore
import math

'''
This is the class for the Gaze Estimation Model.
contains mefthods for loading, checking, and running inference and 
methods to pre-process the inputs and detections to the model
'''

class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None,prob_threshold = 0.60):
        #Initialize any class variables desired ###
        self.device = device
        self.extensions = extensions
        self.model_xml = model_name
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.prob_threshold = prob_threshold
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None

    def load_model(self):
        '''
        This method is helping us to load the model (Ir format).
        '''
        #Initialize the plugin and create the network
        self.plugin = IECore()
        
        # Read the IR as a IENetwork
        try:
            self.network = self.plugin.read_network(model=self.model_xml, weights=self.model_bin)
        except Exception as e:
            raise ValueError("Error on Reading the IR  as a IENetwork !! check path for model name")
        
        # check_model support layers
        self.check_model()
        
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        # Get the input layer
        self.input_blob = [i for i in self.network.inputs.keys()]
        # Return the shape of the input layer ###
        self.input_shape = self.network.inputs[self.input_blob[1]].shape
        # Get the output layer
        self.output_blob = [i for i in self.network.outputs.keys()]
        #Extract and return the output results
        #self.output_shape = self.network.outputs[self.output_blob].shape

    def check_model(self):
        '''
        # Check for supported layers ###
        # Check for any unsupported layers, and let the user 
        # know if anything is missing. Exit the program, if so
        '''
        try:
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers)!=0 and self.device =='CPU':
                    log.error("Unsupported layers found: {}".format(unsupported_layers))
                    log.error("try to add extension")
                    self.plugin.add_extension(self.extensions, self.device)
        except Exception as e:
            log.error('fail to add unsupported layers')
            exit()
        
    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        does predictions from the input.
        '''
        try:
            le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
            outputs = self.exec_network.infer({'head_pose_angles':hpa, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
            mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)
            return mouse_coord, gaze_vector
        except Exception as e:
            log.error('Gaze Estimation failed, no gaze vector detected in the frame')
            exit()

    def preprocess_input(self, left_eye, right_eye):
        ''' 
        Allow us to pre-process the inputs before feeding them into the model.
        '''
        le_image_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        re_image_resized = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
        le_img_processed = np.transpose(np.expand_dims(le_image_resized,axis=0), (0,3,1,2))
        re_img_processed = np.transpose(np.expand_dims(re_image_resized,axis=0), (0,3,1,2))
        return le_img_processed, re_img_processed
            

    def preprocess_output(self, outputs,hpa):
        '''
        The detections of the model need to be preprocessed before they can 
        be used by other models or be used to change the mouse pointer.
        we can do that in here 
        '''
        gaze_vector = outputs[self.output_blob[0]].tolist()[0]
        rollValue = hpa[2] 
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        x = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        y = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (x,y), gaze_vector