import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IECore
from var_dump import var_dump 

'''
This is the class for the Facial landmarks Detection Model.
contains mefthods for loading, checking, and running inference and 
methods to pre-process the inputs and detections to the model
'''
class FacialLandmarksDetection:
    '''
        Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model_name,device='CPU', extensions=None,prob_threshold=0.60,visualflags=None):
        #Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None
        self.device = device
        self.extensions = extensions
        self.model_xml = model_name
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.prob_threshold = prob_threshold
        self.flags = visualflags
    

    def check_model(self):
        # Read the IR as a IENetwork
        try:
            self.network = self.plugin.read_network(model=self.model_xml, weights=self.model_bin)
        except Exception as e:
            raise ValueError(" Facial LandMarks Detection | Error on Reading the IR  as a IENetwork !! check path for model name")
    
    def check_plugin(self) :
        '''
        # Check for supported layers ###
        # Check for any unsupported layers, and let the user 
        # know if anything is missing. Exit the program, if so
        '''
        try:
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers)!=0 and self.device == 'CPU':
                    log.error(" Facial LandMarks Detection | Unsupported layers found: {}".format(unsupported_layers))
                    self.plugin.add_extension(self.extensions, self.device)
        except Exception as e:
            log.error('Facial LandMarks Detection | fail to add unsupported layers')
            exit()
        
    def load_model(self):
        '''
        This method is helping us to load the model (Ir format).
        '''
        #Initialize the plugin and create the network
        self.plugin = IECore()
        # check_model 
        self.check_model()
        # check_model support layers
        self.check_plugin()
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        # Return the shape of the input layer ###
        self.input_shape = self.network.inputs[self.input_blob].shape
        # Get the output layer
        self.output_blob = next(iter(self.network.outputs))
        #Extract and return the output results
        self.output_shape = self.network.outputs[self.output_blob].shape
        
    def predict(self, image):
        '''
        does predictions from the input.
        '''
        #try:
        img_processed = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_blob:img_processed})
        outputs = outputs[self.output_blob]
        l_eye, r_eye = self.preprocess_output(outputs,image)
        return l_eye, r_eye
        #except Exception as e:
            #log.error(' Facial Land Marks | No prediction found responding to the input')
            #exit()

    def preprocess_input(self, image):
        ''' 
        Allow us to pre-process the inputs before feeding them into the model.
        '''
        self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        self.image = self.image.transpose((2,0,1))
        self.image = self.image.reshape(1, *self.image.shape)
        return self.image
            

    def preprocess_output(self, outputs,image):
        '''
        The detections of the model need to be preprocessed before they can 
        be used by other models or be used to change the mouse pointer.
        we can do that in here 
        '''
        outputs = outputs[0]
        
        # collect coord for right eye
        r_eye_xmin = int(outputs[2][0][0] * image.shape[1]) - 10
        r_eye_ymin = int(outputs[3][0][0] * image.shape[0]) - 10
        r_eye_ymax = int(outputs[3][0][0] * image.shape[0]) + 10
        r_eye_xmax = int(outputs[2][0][0] * image.shape[1]) + 10
        r_eye_image = image[r_eye_ymin:r_eye_ymax, r_eye_xmin:r_eye_xmax] # mounted the right eye
        
        # collect coordro for left eye
        l_eye_xmin = int(outputs[0][0][0] * image.shape[1]) - 10
        l_eye_ymin = int(outputs[1][0][0] * image.shape[0]) - 10
        l_eye_xmax = int(outputs[0][0][0] * image.shape[1]) + 10
        l_eye_ymax = int(outputs[1][0][0] * image.shape[0]) + 10
        l_eye_image = image[l_eye_ymin:l_eye_ymax, l_eye_xmin:l_eye_xmax] # mounted the left eye 
        
        try :
            if 'facial_landmark_detection' == self.flags:
                eye_points = [[l_eye_xmin, l_eye_ymin, l_eye_xmax, l_eye_ymax],
                            [r_eye_xmin, r_eye_ymin, r_eye_xmax, r_eye_ymax]]
                cv2.rectangle(image, (eye_points[0][0]-10, eye_points[0][1]-10), (eye_points[0][2]+10, eye_points[0][3]+10),
                        (0, 255, 255), 3)
                cv2.rectangle(image, (eye_points[1][0]-10, eye_points[1][1]-10), (eye_points[1][2]+10, eye_points[1][3]+10),
                        (0, 255, 255), 3)
        except Exception as e :
             log.error(" Facial LandMarks Detection | Could not detect eyes  points in the frame, Index is out of range")
             exit()                  
        return l_eye_image, r_eye_image