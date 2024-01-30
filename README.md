# Visual Object Tracking Project

This project encompasses a series of object tracking implementations, each corresponding to a different part of the project (TP1, TP2, TP3, and TP4). Below are the descriptions and execution instructions for each part.

## TP1: Basic Object Tracker

### Description
This part of the project implements a basic object tracking system. It uses the OpenCV library to read a video file and the Detector class to detect objects in each frame. The tracker uses the detected bounding boxes to track the objects across frames.

### Execution
To run the basic object tracker, use the following command:

```bash
python3 objTracker.py 'path_to_the_video'
```

Replace `'path_to_the_video'` with the path to your video file.

## TP2: Greedy Tracker

### Description
TP2 introduces a greedy tracking algorithm that uses Intersection over Union (IoU).

### Execution
To run the greedy tracker, execute:

```bash
python3 greedyTracker.py
```

## TP3: Hungarian Tracker

### Description
TP3 implements the Hungarian algorithm for object tracking using IoU and visual similarity.

### Execution
To run the Hungarian tracker, use the following command:

```bash
python3 hungarianTracker.py
```


## TP4: Kalman Tracker

### Description
In TP4, the Kalman Filter is integrated into the tracking system for improved accuracy and robustness.

### Execution
To execute the Kalman Tracker, run:

```bash
python3 TP4.py
```