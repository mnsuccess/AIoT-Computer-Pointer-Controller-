from argparse import ArgumentParser

from mouse_controller import MouseController
from input_feeder import InputFeeder

from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from facial_landmarks_detection import FacialLandmarksDetection

import cv2
import os
import time
import logging as log
import numpy as np

def file_not_found(self,filename):
    log.error(" xml "+self.fileName+" file not found")
    exit(1)
        

def main(args):
    
    #initailizing  video input argurment get feed
    video_input = args.video
    feed = None
    video_type = None
    if video_input != "cam":
        if not os.path.isfile(video_input):
            log.error("Unable to find specified video file")
            exit(1)
        video_type = 'video'
        feed = InputFeeder(input_type = video_type,input_file = video_input)
    else:
        video_type = 'cam'
        feed = InputFeeder(input_type = video_type,input_file = video_input)

    # initialise timer for model loading
    model_load_time = time.time()
    
    # initializing the mousecontroller
    mouseController = MouseController('medium','fast')
    
    #initailizing models
    if not os.path.isfile(args.modelfaciallandmark) :
        file_not_found(args.modelfaciallandmark)
    elif not os.path.isfile(args.modelgazeestimation):
        file_not_found(args.modelgazeestimation)
    elif not os.path.isfile(args.modelfacedetection):
        file_not_found(args.modelfacedetection)
    elif not os.path.isfile(args.modelheadpose):
        file_not_found(args.modelheadpose)
    
    if 'face_detection' in args.visualFlags : 
        faceDetection = FaceDetection(args.modelfacedetection, args.device, args.cpu_extension,args.prob_threshold,visualflags='face_detection')
    else:
        faceDetection = FaceDetection(args.modelfacedetection, args.device, args.cpu_extension,args.prob_threshold)
        
    if 'facial_landmark_detection' in args.visualFlags : 
        facialLandMaskDetection = FacialLandmarksDetection(args.modelfaciallandmark, args.device, args.cpu_extension ,visualflags='facial_landmark_detection')
    else:
        facialLandMaskDetection = FacialLandmarksDetection(args.modelfaciallandmark, args.device, args.cpu_extension)
    
    if 'pose_estimation' in args.visualFlags : 
        headPoseEstimation = HeadPoseEstimation(args.modelheadpose, args.device, args.cpu_extension,visualflags='pose_estimation')
    else:
        headPoseEstimation = HeadPoseEstimation(args.modelheadpose, args.device, args.cpu_extension)
        
    if 'gaze_estimation' in args.visualFlags :  
        gazeEstimation = GazeEstimation(args.modelgazeestimation, args.device, args.cpu_extension,visualflags='gaze_estimation')
    else:
        gazeEstimation = GazeEstimation(args.modelgazeestimation, args.device, args.cpu_extension)
    
    # Loading models
    faceDetection.load_model()
    facialLandMaskDetection.load_model()
    headPoseEstimation.load_model()
    gazeEstimation.load_model()
    all_model_load_time = time.time() - model_load_time
    
    
    # start inference timer and Init counter
    inference_time = time.time()
    counter = 0
    feed.load_data()
    
    for ret, frame in feed.next_batch():
        if frame is None:
            log.error('The file is corrupted!! no input frame was being read')
            exit()
        if not ret:
            break
        counter+=1
        #  here we run mutilple models prediction
        frame,face_crop, detections = faceDetection.predict(frame)
        left_eye, right_eye = facialLandMaskDetection.predict(face_crop)
        headpose_out = headPoseEstimation.predict(face_crop, frame)
        new_mouse_coord, gaze_vector = gazeEstimation.predict(left_eye, right_eye, headpose_out,frame)
        
        # add inference time text
        total_time = time.time() - inference_time
        inf_time_message = "Manasse_Ngudia | Inference time: {:.3f}ms"\
                               .format(total_time  * 1000)
        cv2.putText(frame, inf_time_message, (15, 50),cv2.FONT_HERSHEY_COMPLEX, 1, (200, 10, 10), 3)
        #Display frame
        cv2.imshow('frame', frame)
        #Controlle the  mouse
        mouseController.move(new_mouse_coord[0],new_mouse_coord[1])
        # Break if escapturee key pressed
        if video_type != 'video':
            tm=1
        else:
            tm=500
        key_pressed = cv2.waitKey(tm)
        if key_pressed == 27:
            break
        
    log.error("VideoStream  closed...")
    cv2.destroyAllWindows()
    feed.close()
    
    total_time = time.time() - inference_time
    total_inference_time = round(total_time, 1)
    fps = counter / total_inference_time
    
    print("Total time for loading all the models was :"+str(all_model_load_time)+"secondes")
    
    print("Total inference time of the models was :"+str(total_inference_time)+"secondes")
    
    print("Total number of frames per second was :"+str(fps)+"fps")

if __name__ == '__main__':
    #Parse command line arguments.

    #:return: command line arguments
    parser = ArgumentParser()
    parser.add_argument("-mfd", "--modelfacedetection", required=True, type=str,
                        help="Path to .xml file of  pretrained Face Detection model.")
    parser.add_argument("-mfl", "--modelfaciallandmark", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-mhp", "--modelheadpose", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-mge", "--modelgazeestimation", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-v", "--video", required=True, type=str,
                        help="Path of video file or enter cam for webcam.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for detections filtering (0.6 by default)")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-flags", "--visualFlags", required=False, nargs='+',
                        default=[],
                        help = "Visualizing  of the output model ."
                        "for Face Detection model output, Enter face_detection "
                        "for Facial Landmark Detection model, Enter facial_landmark_detection "
                        "for Head Pose Estimation model, Enter pose_estimation "
                        "for Gaze Estimation model , Enter gaze_estimation "
                        "example -flags face_detection facial_landmark_detection pose_estimation gaze_estimation (Space separated if multiple values)")    
    args = parser.parse_args()
    main(args) 