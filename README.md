# Computer Pointer Controller
A Computer Pointer Controller allows user to control the mouse movements with their eye-gaze which will be captured through a webcam or even a video by using OpenVINO Toolkit along with its Pre-Trained Models which helps to deploy AI at Edge. This project run with multiple models in the same machine and coordinate the flow of data between models.

## Project Set Up and Installation
Before you start running this project, you'll need to get your local environment set up on your own machine. Here are the main things to do:
- Download and install the OpenVINO Toolkit. The installations directions for OpenVINO can be found [here](https://docs.openvinotoolkit.org/latest/index.html)
- Run the Verification Scripts to verify your installation. This is a very important step to be done before you proceed further.
- The  Download Pre-Trained models from OpenVINO Open model zoo using the ```model downloader``` are:
    1. [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
    2. [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
    3. [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
    4. [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
 These models have already be dowloaded and stored in Intel folder.

 -- The project directory is structured as follows:
```
    project
    |  
    |_ bin
    |  |_demo.mp4
    |  |_demo_cam.png
    |  |_demo_video.png
    |- intel
    |  |All model must be here      
    |_ README.md    
    |   
    |_ requirements.txt   
    |    
    |_src
        |_ main.py
        |_ input_feeder.py
        |_ mouse_controller.py
        |_ face_detection.py
        |_ head_pose_estimation.py
        |_ facial_landmarks_detection.py
        |_ gaze_estimation.py
``` 
## Demo
[![Demo video](https://img.youtube.com/vi/t8uR_jaJIzY/0.jpg)](https://youtu.be/t8uR_jaJIzY)
#### Step 1
Clone the repository:- https://github.com/mnsuccess/AIoT-Computer-Pointer-Controller.git
Open a new terminal and run the following commands:-
#### Step 2
Change the directory to src directory of project repository
``` 
cd <path_to_project_directory>/src
```
#### Step 3
 Now, run the following command to run our application
```
python3 main.py -mdf <path_to_project_directory>/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml   -mfl <path_to_project_directory>/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -mhp  <path_to_project_directory>/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml  -mge  <path_to_project_directory>/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -v <path_to_project_directory>/bin/demo.mp4
```

## Documentation
- The ```python main.py -h``` command displays the commands which are supported by project:
  usage: main.py [-h] -mfd MODELFACEDETECTION -mfl MODELFACIALLANDMARK -mhp
               MODELHEADPOSE -mge MODELGAZEESTIMATION -v VIDEO
               [-prob PROB_THRESHOLD] [-d DEVICE] [-l CPU_EXTENSION]
               [-flag VISUALFLAG]

optional arguments:
  -h, --help            show this help message and exit
  -mfd MODELFACEDETECTION, --modelfacedetection MODELFACEDETECTION
                        Path to .xml file of pretrained Face Detection model.
  -mfl MODELFACIALLANDMARK, --modelfaciallandmark MODELFACIALLANDMARK
                        Specify Path to .xml file of Facial Landmark Detection
                        model.
  -mhp MODELHEADPOSE, --modelheadpose MODELHEADPOSE
                        Specify Path to .xml file of Head Pose Estimation
                        model.
  -mge MODELGAZEESTIMATION, --modelgazeestimation MODELGAZEESTIMATION
                        Specify Path to .xml file of Gaze Estimation model.
  -v VIDEO, --video VIDEO
                        Path of video file or enter cam for webcam.
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering (0.6 by
                        default)
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -flag VISUALFLAG, --visualflag VISUALFLAG
                        Visualizing of the output model .for Face Detection
                        model output, Enter face_detection for Facial Landmark
                        Detection model, Enter facial_landmark_detection for
                        Head Pose Estimation model, Enter pose_estimation for
                        Gaze Estimation model , Enter gaze_estimation

## Benchmarks
The benchmark result of running my model on **CPU** with multiple model precisions are :
- INT8:
  - The total model loading time is : 5.13sec
  - The total inference time is : 7.9sec
  - The total FPS is : 0.42fps

- FP16:
  - The total model loading time is : 1.72sec
  - The total inference time is : 7.8sec
  - The total FPS is : 0.45fps 

- FP32:
  - The total model loading time is : 3.451sec
  - The total inference time is : 12.4sec
  - The total FPS is : 0.36fps
  
The benchmark result of running my model on **IGPU[Intel HD Graphics 630]** with multiple model precisions are :
- FP32:
  - The total model loading time is : 59.797sec
  - The total inference time is : 8.67sec
  - The total FPS is : 0.3998fps
  
- FP16:
  - The total model loading time is : 67.3sec
  - The total inference time is : 9.2sec
  - The total FPS is : 0.425fps

## Results
- With the above benchmark results. Faster Inference is obtaned using less precision model.
by reducing the precision, the usage of memory is less and its less computationally expensive when compared to higher precision models. 
- comparing the results between FP16 and INT8, the inference is same but the model loading time was more.


## Stand Out Suggestions
- I improved my model inference time by changing the precisions of the models,


### Edge Cases

1. If there are Multiple face detected in the frame then model takes the first detected face for control the mouse  pointer.

