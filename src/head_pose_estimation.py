import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IECore

'''
This is the class for a Head Pose Estimation Model.
contains mefthods for loading, checking, and running inference and 
methods to pre-process the inputs and detections to the model
'''
class HeadPoseEstimation:
    '''
    Class for the HeadPoseEstimation Model.
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