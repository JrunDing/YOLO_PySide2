# YOLOv5  PySide2         Object  Detection UI 

## Introduction

​	This project uses pyside2 to create UI and execute yolov5 object detection alogrithm.

​	After installing dependency, you can execute `yolov5ui.py` to view the result.

​	I dont transform `.ui` file to python code. Of course, you can execute `pyside2-uic main.ui > ui_main.py` to get python code if you want that.

## Dependency

​	python == 3.7.12

​	numpy == 1.21.6

​	opencv-python == 4.1.1.26

​	opencv-contrib-python == 4.4.0.44

​	Pillow == 9.2.0

​	PySide2 == 5.15.2.1

​	torch == 1.7.0

​	torchvision == 0.8.0

​	cudatoolkit == 10.2.89

​	tqdm == 4.64.0

## How to use

​	Firstly you need to install dependency above. 

​	Secondly you need to download YOLOv5 model from [YOLOv5 official](https://github.com/ultralytics/yolov5). 

​	Finally you need to prepare some pictures and videos.

​	Directly execute `yolov5ui.py` to view result.

## Demo Video

​	https://www.bilibili.com/video/BV1wk4y1b77c/?spm_id_from=333.999.0.0

## Reference

[1] Zaidi S S A, Ansari M S, Aslam A, et al. A survey of modern deep learning based object detection models[J]. Digital Signal Processing, 2022: 103514.

[2] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

[3] Bochkovskiy A, Wang C Y, Liao H Y M. Yolov4: Optimal speed and accuracy of object detection[J]. arXiv preprint arXiv:2004.10934, 2020.

https://github.com/ultralytics/yolov5

https://github.com/bubbliiiing/yolov5-pytorch

