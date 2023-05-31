# Camera Pose estimation from a video

## Required libraries

please make sure to install following before executing the above code
```python
pip3 install matplotlib
pip3 install numpy
pip3 install scipy
pip3 install scikit-learn
pip3 install opencv-python==4.6.0
```
## Instructions to run the code
    Run "python3 Problem1.py"

## Implementation
The camera pose estimation from video is implemented as shown in the below pipeline

 ![Alt text](./doc/images/pipeline.png?raw=true "pipeline")

####  Choice of Real world coordinates:

* Real world axis is chosen at the center of the page with x-axis parallel to the
longer edge and y-axis parallel to the shorter edge

## Results
* Computed Camera Pose Estimation for over 150 frames of the given videos is as shown below. The code would take upto ~1min 10sec to process the video.


 ![Alt text](./doc/images/poses.png?raw=true "estimated poses")

* Hough-lines based corner detection.
<video width="640" height="360" controls>
  <source src="https://github.com/itej89/CameraPose_HoughTransform/assets/37236721/26c36835-9b6b-4061-9813-474db96e4785" type="video/mp4">
</video>



