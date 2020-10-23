# Computer Vision Pipeline
 A Deep Learning Computer Vision Pipeline for Object Detection and Classification. Computer vision has become an integral part of artificial intelligence and their applications are exponentially growing in the industry. They are progressing in many domains and one such domain is object detection. Given a set of object classes, the goal of object detection is to identify the scale and position of object instances in an image, if any. After detecting the object instances in the image, the information can be further processed to acquire more details about that object (color, shape, size, etc.). This assignment implements the principles of object detection to identify the number of cars, type of car(s) (sedan or hatchback) and the color of the car(s) in each frame of a video. This is done using Tiny-YOLO which is a variation of the YOLO architecture. This report addresses the details of the implementation of the program by providing details on the pipeline design approach and decisions, training of the car type classifier model, the pipeline outputs and the strengths and weaknesses of the design along with their prospects. 
 
# Requirments
Tensorflow version = 1.15
Keras Version = 2.3.0

Initializing Tiny YOLO model for faster processing speed 

[1] Download the tiny weights from - https://pjreddie.com/media/files/yolov3-tiny.weights

[2] python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolo.h5

In the yolo.py file 
[3] Under the default function
	Change the achors from "yolo_anchors.txt" to "tiny_yolo_anchors.txt"

Now the Tiny yolo model is ready to be used 


-> myPipeline.py is the main file
-> car_type_classifier.py contains the classification model
-> Colour_detectcor.py contains the color classifier

In car_type_classifier change the training image destination (in both training and validation)


# Steps to run:

In the console>>


* "python myPipeline.py 1"    For executing query 1
* "python myPipeline.py 2"    For executing query 1 and 2
* "python myPipeline.py 3"    For executing query 1,2 and 3
 

Note:
-> The graph for all the three queries is saved in the root folder in PNG format.
-> Executing the last query will give the final F1-score. 
-> If you run query 3 only then resulting video will be saved.
-> As the classifier uses saved model to classify the car, after running 
	the program once, the model is trained, we can comment 
	the line 314 "car_Type_model()" in myPipeline.py. So that we do not have to
	retrain the model again and again.
