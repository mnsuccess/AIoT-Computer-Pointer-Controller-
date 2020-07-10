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
    def __init__(self, model_name, device='CPU', extensions=None,prob_threshold=0.60,visualflags=None):
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
            raise ValueError("Error on Reading the IR  as a IENetwork !! check path for model name")
    
    def check_plugin(self) :
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
                    self.plugin.add_extension(self.extensions, self.device)
        except Exception as e:
            log.error('fail to add unsupported layers')
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
