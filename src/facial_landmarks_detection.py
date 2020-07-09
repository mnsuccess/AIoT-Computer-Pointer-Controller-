import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IECore

'''
This is the class for the Facial landmarks Detection Model.
contains mefthods for loading, checking, and running inference and 
methods to pre-process the inputs and detections to the model
'''
class FacialLandmarksDetection:
    '''
        Class for the Facial Landmarks Detection Model.
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
        self.input_blob = next(iter(self.network.inputs))
        # Return the shape of the input layer ###
        self.input_shape = self.network.inputs[self.input_blob].shape
        # Get the output layer
        self.output_blob = next(iter(self.network.outputs))
        #Extract and return the output results
        self.output_shape = self.network.outputs[self.output_blob].shape

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
        
    def predict(self, image):
        '''
        does predictions from the input.
        '''
        try:
            img_processed = self.preprocess_input(image.copy())
            outputs = self.exec_network.infer({self.input_blob:img_processed})
            points = self.preprocess_output(outputs)
            height=image.shape[0]
            widht=image.shape[1]
            points = points* np.array([widht, height, widht, height])
            points = points.astype(np.int32) 
            le_xmin=points[0]-10
            le_ymin=points[1]-10
            le_xmax=points[0]+10
            le_ymax=points[1]+10
            re_xmin=points[2]-10
            re_ymin=points[3]-10
            re_xmax=points[2]+10
            re_ymax=points[3]+10
            left_eye =  image[le_ymin:le_ymax, le_xmin:le_xmax]
            right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
            eye_points = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]
            return left_eye, right_eye, eye_points
        except Exception as e:
            log.error('No prediction found responding to the input')
            exit()

    def preprocess_input(self, image):
        ''' 
        Allow us to pre-process the inputs before feeding them into the model.
        '''
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_cvt, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return img_processed
            

    def preprocess_output(self, outputs):
        '''
        The detections of the model need to be preprocessed before they can 
        be used by other models or be used to change the mouse pointer.
        we can do that in here 
        '''
        results = outputs[self.output_blob][0]
        reye_y = results[3].tolist()[0][0]
        reye_x = results[2].tolist()[0][0]
        leye_y = results[1].tolist()[0][0]
        leye_x = results[0].tolist()[0][0]
        return (leye_x, leye_y, reye_x, reye_y)