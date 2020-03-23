# donttouchyourface

- The things you need to use it

  - YOLOv3
    - Your own data set to train the model
  - OpenCV
  - PyGame
  - Camera

A quick summary of what I did:

- Step 1- Created a small dataset which contains images of my face and hands
- Step 2- Trained a YOLOv3 model using the dataset
- Step 3- After 3 hours of training, the model learned to detect and localize my face and hands in an image
- Step 4- Used the AWS Polly speech synthesis tool to create a few funny audio files to warn my self
- Step 5- Finally, I created a Python script that captures images from my webcam uses the pre-trained model for inference and warns me if my hand gets too close to my face

![Alt Text](face.png)

Test System:
- GPU: Nvidia GTX 1050Ti 4GB
- CPU : Intel i5-3470
- RAM: 12GB
- OS: Ubuntu 18.04

Detailed steps:

- Step 1- Create a small dataset which contains images of your face and hands
  - 1.A: Install cheese to take pictures
    - sudo apt install cheese
  - 1.B: Take pictures of your hands and face
    - I took 30 hand pictures and 30 face pictures. The more the better!
  - 1.C: Label your data set using labelImg tool
    - There are several different tools to do this step. I used https://github.com/tzutalin/labelImg

- Step 2- Trained a YOLOv3 model using the dataset
  - 2.A: Install Darknet: https://pjreddie.com/darknet/install/
      - Install CUDA https://developer.nvidia.com/cuda-downloads
  - 2.B: Download pre-trained model trained on Imagenet datase becasue it is better starting with some pre-trained weigths than starting with random weights.
    - wget https://pjreddie.com/media/files/darknet53.conv.74
  - 2.C: Train a model using your data set: https://pjreddie.com/darknet/yolo/
    - It took about 3 hours to start getting good results.
      - I overcloked my GPU to save time using the Green with Envy tool (Optional)
       - https://gitlab.com/leinardi/gwe
       - I am not sure how much time it saved me :/ I probably spent more time on learning how to overclock my GPU :)
    - ./darknet detector train cfg/obj.data cfg/yolov3-voc.cfg darknet53.conv.74
      - cfg/obj.data: The file tells where to find your dataset, class names and trained models
      - cfg/yolov3-voc.cfg: YOLO network architecture configuration. 
        - I had to reduce the batchsize and number of subdivision becasue my GPU has only 4 GB of memery.
          - batch=16
          - subdivisions=16
      - darknet53.conv.74: pre-trained weights
      
  - Step 3- After 3 hours of training, the model shoudl learn to detect and localize your face and hands in an image
    -3.A: Find the model in /backup
    -3.B: Test the model
        
  - Step 4- Use the AWS Polly speech synthesis tool to create a few funny audio files to warn my self. I uplaeded the audio files if you want to skip this part.
    -4.A: You have to create an AWS account.
    -4.B: https://aws.amazon.com/polly/
      
 - Step 5- Finally, Create a Python script that captures images from my webcam uses the pre-trained model for inference and warns me if my hand gets too close to my face
   -5A: Install Anaconda (Optional)
    - Create a virtual enviroment (Python 3.6) with the following packages installed
      - OpenCV
      - Numpy
      - PyGame (Used it to play audio files
        - python3 -m pip install -U pygame --user
   -5B: You can either use the Python wrapper that I created for YOLOv3 and write your own application or use the corona.py as an example.
     -  https://github.com/mlherd/darknet/tree/python36_wrapper
     - In order to detect touches, I used rectangle overlap algorithm to check if a hand overlaps with a face.
    
