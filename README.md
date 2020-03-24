# donttouchyourface

- The things you need to build your own

  - YOLOv3
    - Your data set to train the model
  - OpenCV
  - PyGame
  - Camera

A quick summary of what I did:

- Step 1- Created a small dataset which contains images of my face and hands
- Step 2- Trained a YOLOv3 model using the dataset
- Step 3- After 3 hours of training, the model learned to detect and localize my face and hands in an image
- Step 4- Used the AWS Polly speech synthesis tool to create a few funny audio files to warn my self
- Step 5- Finally, I created a Python script that captures images from my webcam, uses the pre-trained model for inference and warns me if my hand gets too close to my face

![Alt Text](face.png)

My ancient testing system:
- GPU: Nvidia GTX 1050Ti 4GB
- CPU: Intel i5-3470
- RAM: 12GB
- OS: Ubuntu 18.04

Detailed steps:

- Step 1- Create a small dataset which contains images of your face and hands
  - 1.A: Install cheese to take pictures
    - sudo apt install cheese
  - 1.B: Take pictures of your hands and face
    - I took 30 hand pictures and 30 face pictures. Remember the more the better!
      - 20 for training and 10 for testing for each class.
      - You will have all your pictures in one folder, so make sure the dataset is shuffled.
  - 1.C: Label your data set using the labelImg tool
    - There are several different tools to do this step. I used https://github.com/tzutalin/labelImg

- Step 2- Train a YOLOv3 model using your dataset
  - 2.A: Install Darknet: https://pjreddie.com/darknet/install/
      - Also, Install CUDA https://developer.nvidia.com/cuda-downloads. Trust me, you want to use an Nvidia GPU with CUDA to do this.
  - 2.B: Download the pre-trained model (darknet53.conv.74) which is trained on Imagenet dataset because it is better to begin with some pre-trained weights than complete random weights.
    - wget https://pjreddie.com/media/files/darknet53.conv.74
  - 2.C: Train a model using your data set: https://pjreddie.com/darknet/yolo/
    - It took about 3 hours to start getting good results.
      - I overclocked my GPU to save time using the Green with Envy tool (Optional)
       - https://gitlab.com/leinardi/gwe
       - I am not sure how much time overclocking saved me. I probably spent more time on learning how to overclock my GPU :/
    - Training
      - ./darknet detector train cfg/obj.data cfg/yolov3-voc.cfg darknet53.conv.74
        - cfg/obj.data: The file tells where to find your dataset, class names and trained models
        - cfg/yolov3-voc.cfg: YOLO network architecture configuration. 
          - I had to reduce the batch size and the number of subdivisions because my GPU has only 4 GB of memory.
            - batch=16
            - subdivisions=16
        - darknet53.conv.74: pre-trained weights
        
- Step 3- After 3 hours of training, the model should learn to detect and localize your face and hands in an image.
  - 3.A: Find the model in /backup
  - 3.B: You will need a webcam for testing.
  - 3.C: Test the model
    - You can use the YOLO demo to quick test the accuracy of your model.
    - ./darknet detector demo cfg/obj.data cfg/yolov3-voc_inf.cfg yolov3-voc.backup
      - yolov3-voc.backup: Your model
      - cfg/yolov3-voc_inf.cfg: We need to comment on the training config lines and uncomment testing config lines. In other words, you need to make sure, the batch size and the number of subdivision values are set to 1.
        - batch=1
        - subdivisions=1

- Step 4- Use the AWS Polly speech synthesis tool to create a few funny audio files to warn my self. I uploaded the audio files that I created if you want to skip this part.
  - 4.A: You have to create an AWS account.
  - 4.B: https://aws.amazon.com/polly/

- Step 5- Finally, create a Python script that captures images from my webcam uses the pre-trained model for inference and warns me if my hand gets too close to my face
  - 5A: Install Anaconda (Optional)
    - Create a virtual environment (Python 3.6) with the following packages installed
      - OpenCV
      - Numpy
      - PyGame (Used it to play audio files
      - python3 -m pip install -U pygame --use
  - 5B: You can either use the Python wrapper that I created for YOLOv3 and write your own application or use the corona.py as an example.
      -  https://github.com/mlherd/darknet/tree/python36_wrapper
      - To detect touches, I used the rectangle overlap algorithm to check if a hand overlaps with a face.
    
Good luck and stay sterilized!
