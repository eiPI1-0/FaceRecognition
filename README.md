# FaceRecognition with C++ & OpenCV

## Requirements

- Any x86_64 or Arm Platform
- OpenCV 4.5.1
- CMake
- Make
- g++

Test on openSUSE Leap 15.2 x86_64 with Intel Core i7-7700HQ, Fedora with EAIDK 310 and EAIDK 610.

## Registration

Path: `FaceRecognition/data/`

The registered user's data is a subfolder named by user name and filled with their photos. The system's performance depends on the number of each user's photos, and the accuracy depends on the consistency between the user's registered environment and the recognition environment.

## Model

Path: `FaceRecognition/models/`

Use https://github.com/ShiqiYu/libfacedetection.train/blob/master/tasks/task1/onnx/YuFaceDetectNet.onnx as detection model by default.

Use https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7 as extraction model by default.

## Compiling

```sh
cd FaceRecognition
cmake .
make
```

## Running

```sh
./main 0
```

where 0 is the camera ID.
