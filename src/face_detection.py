import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

'''
This is the class for a FaceDetection Model.
contains mefthods for loading, checking, and running inference and 
methods to pre-process the inputs and detections to the model
'''

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None,prob_threshold=0.60):
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
            results = self.exec_network.infer({self.input_blob:img_processed})
            detections = results[self.output_blob]
            points = self.preprocess_output(detections)
            points = points[0] 
            height=image.shape[0]
            width=image.shape[1]
            points = points* np.array([width, height, width, height])
            points = points.astype(np.int32)
            cropped_face = image[points[1]:points[3], points[0]:points[2]]
            return cropped_face, points
        except Exception as e:
            log.error('Face detection failed, no face detected in the frame')
            exit()

    def preprocess_input(self, image):
        ''' 
        Allow us to pre-process the inputs before feeding them into the model.
        '''
        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        #img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        image_processed = image_resized.transpose((2,0,1))
        image_reshaped = image_processed.reshape(1, *image_processed.shape)
        return image_reshaped

    def preprocess_output(self, detections):
        '''
        The detections of the model need to be preprocessed before they can 
        be used by other models or be used to change the mouse pointer.
        we can do that in here 
        '''
        points =[]
        outs = detections[0][0]
        for box in outs:
            conf = box[2]
            if conf>self.prob_threshold:
                x_min=box[3]
                y_min=box[4]
                x_max=box[5]
                y_max=box[6]
                points.append([x_min,y_min,x_max,y_max])
        return points
