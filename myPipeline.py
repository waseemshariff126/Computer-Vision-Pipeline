# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:28:56 2020

@author: waseem
"""

#PIPELINE IMPORTS-------------------
import concurrent.futures
import multiprocessing as mp
import time
import copy
import sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
#-----------------------------------


#MODEL IMPORTS----------------------
import cv2
from PIL import Image
import sys
from car_type_classifier import car_type_model
from Colour_detector import colorDetector
from yolo import YOLO
import tensorflow as tf
from tensorflow import keras 
#----------------------------


#Query Option (Choose Qeury to execute (between 1 to 3))
queryOption = int(sys.argv[1])
print(f'You have selected - {queryOption}')


#Calculating Time Stats:
q1Time = 0
q2Time = 0
q3Time = 0
stat1List = []
stat2List = []
stat3List = []
processedFrameList = []
frameNumList = []

#Clearing the CSV file for results of the current run:
fname = 'ourResult.csv'
f = open(fname , "w+")
f.close()

#Storing the results to compare with ground truth-------------------------------
initialCount = {
    'carCount' : 0,
    'Sedan' : {
        'Black' : 0,
        'Silver' : 0,
        'Red': 0,
        'White': 0,
        'Blue': 0},
    'Hatchback' : {
         'Black' : 0,
        'Silver' : 0,
        'Red': 0,
        'White': 0,
        'Blue': 0}
    }
#-------------------------------------------------------------------------------

#----------------------------------PLOTTING STATS GRAPH-------------------------
def statsGraph():

    processedFrameList = np.load('ProcessedFrames.npy')
    frameNumList = np.load('frameNumList.npy')
    frameNumList = np.delete(frameNumList, 0)

    #For Stats 1
    if queryOption in [1,2,3]:
        stat1List = np.load('stats1.npy')
        stat1List = np.delete(stat1List, 0)
        plt.plot(frameNumList, stat1List, color = 'red')
        plt.title('Stats 1 Graph')
        plt.xlabel('Processed Frame Number' )
        plt.ylabel('Time (seconds)')
        plt.savefig('Stats 1 Plot.png')
        #plt.show()

    
    #For Stats 2:
    if queryOption in [2,3]:
        processedFrameList = np.delete(processedFrameList, 0)
        stat2List = np.load('stats2.npy')
        stat2List = np.delete(stat2List,0)
        plt.plot(processedFrameList, stat2List, color = 'blue')
        plt.title('Stats 2 Graph')
        plt.xlabel('Processed Frame Number' )
        plt.ylabel('Time (seconds)')
        plt.savefig('Stats 2 Plot.png')
        #plt.show()

    #For Stats 3:
    if queryOption in [3]:
        stat3List = np.load('stats3.npy')
        stat3List = np.delete(stat3List, 0)
        plt.plot(processedFrameList, stat3List, color = 'green')
        plt.title('Stats 3 Graph')
        plt.xlabel('Processed Frame Number' )
        plt.ylabel('Time (seconds)')
        plt.savefig('Stats 3 Plot.png')
        plt.show()
#--------------------------------------------------------------------------

#-----------------------------CALCULATING F1 SCORE-------------------------
def calF1Score():
    #Importing the ground truth and our result two calculate F1 Score
    groundTruth = pd.read_excel('Ground Truth (Assignment 2).xlsx')
    ourResult = pd.read_csv('ourResult.csv', header = None)

    if queryOption in [1,2,3]:
        #Calculating F1 Score for the total number of cars detectd in each frame
        q1GroundTruth = q1GroundTruth = groundTruth.iloc[1: , 11].values
        q1Result =  ourResult.iloc[:, 11].values
        # print('The Confusion Matrix for car count is', confusion_matrix(q1GroundTruth, q1Result))
        print('The f1 Score for car count is', f1_score(q1GroundTruth, q1Result, average = 'macro', zero_division= 0))

    if queryOption in [3]:
        gtType = groundTruth[groundTruth['Total Cars'] != 0]
        gtType = gtType.iloc[1: , 0:]

        cname = ['FrameNum', 'SedanBlack','SedanSilver','SedanRed','SedanWhite','SedanBlue', 'HatchBlack','HatchSilver','HatchRed','HatchWhite','HatchBlue', 'Total Cars']
        gtType.columns = cname

        orType = ourResult
        orType.columns = cname

        finalType = orType.merge(gtType, on = 'FrameNum', how = 'inner')
        finalType.iloc[: , [0, 11,22]].values.tolist()

        gtSedan = finalType.iloc[: , 1:6].values
        gtHatch = finalType.iloc[: , 6:11].values
        gtSedanSum = np.sum(gtSedan, axis = 1).astype(int)
        gtHatchSum = np.sum(gtHatch, axis = 1).astype(int)

        orSedan = finalType.iloc[: , 12:17].values
        orSedanSum = np.sum(orSedan, axis = 1 ).astype(int)
        orHatch = finalType.iloc[: , 17:22].values
        orHatchSum = np.sum(orHatch, axis = 1).astype(int)

        f1ScoreSedan = f1_score(gtSedanSum, orSedanSum, average = 'micro')
        f1ScoreHatch = f1_score(gtHatchSum, orHatchSum, average = 'micro')
        avgf1ScoreForType = (f1ScoreSedan + f1ScoreHatch)/2
        print('The F1 Score for Query 2 (Car Type) is - ', avgf1ScoreForType)

    if queryOption in [3]:
        black = f1_score(finalType.iloc[: , 1].astype(int), finalType.iloc[: , 17].astype(int), average = 'macro')
        silver = f1_score(finalType.iloc[: , 2].astype(int), finalType.iloc[: , 18].astype(int), average = 'macro')
        red = f1_score(finalType.iloc[: , 3].astype(int), finalType.iloc[: , 19].astype(int), average = 'macro')
        white = f1_score(finalType.iloc[: , 4].astype(int), finalType.iloc[: , 20].astype(int), average = 'macro')
        blue = f1_score(finalType.iloc[: , 5].astype(int), finalType.iloc[: , 21].astype(int), average = 'macro')
        avgf1ScoreForColor = (black + silver + red + white + blue) / 5
        print('The F1 Score for Query 3 (Color + Car Type) is - ', avgf1ScoreForColor)
        

#-------------------------------------------------------------------------------

#----------------------GETTING REGION OF INTEREST-------------------------------
def region_of_interest(frame, yolo):
    image = Image.fromarray(frame, 'RGB')
    raw_image, roi, time_value = yolo.detect_image(image)
    BB = []
    for BBox in roi:
        if BBox['class'] == 'car':
            print('Car detected')
            top = BBox['top']
            left = BBox['left']
            bottom = BBox['bottom']
            right = BBox['right']
            car_box = frame[top-20:bottom, left:right+50]
            BB.append([car_box,(top,left,bottom,right)])
    return BB
#-------------------------------------------------------------------------------

# Preprocess the testing image, such that the dimensions of testing image matches training image dimensions
def preprocess_image(img):
    if (img.shape[0] != 180 or img.shape[1] != 180):
        img = cv2.resize(img, (180, 180), interpolation=cv2.INTER_NEAREST)
    img = np.expand_dims(img, axis=0)
    return img  
#------------------------------------------------------------------------------

# As we used sigmoid as our last layer, we fix the threshold classify the car
def car_type_classifier(model, frame):
    pred = model.predict(preprocess_image(frame))
    if pred[0] > 0.5:
        return "Sedan"
    else:
        return "Hatchback"
#------------------------------------------------------------------------------

#Processing the video to get each frame----------------------------------------
def getFrame(qu):
    vidCap = cv2.VideoCapture('video.mp4')
    while(vidCap.isOpened()):    
        success,frame = vidCap.read()
        if success:
            qu.put(frame)
        else: 
            qu.put('end')
            break
#-------------------------------------------------------------------------------


#Processing the frames to get the queries---------------------------------------
def processFrame(qu):
    global q1Time, q2Time, q3Time

    model = keras.models.load_model('callback_model') 
    yolo = YOLO()    
    img_list =[]
    frameNum = 1
    frameProcess = 1
    while True:
        if not qu.empty():
            startTime = time.time()
            finalResult = copy.deepcopy(initialCount)
            x = qu.get()

            # WHEN THE QUEUE IS OVER--------------------------
            if 'end' in x:
                print('The video has ended')
                np.save('stats1.npy', stat1List)
                np.save('stats2.npy', stat2List)
                np.save('stats3.npy', stat3List)
                np.save('ProcessedFrames.npy', processedFrameList)
                np.save('frameNumList.npy', frameNumList)
                break
            #-------------------------------------------------

            else:

                print(f'\nFrame Number - {frameNum}')
                if queryOption in [1,2,3]: 

                    # Query 1 : To Find the cars in the frame and take the region of intertest----------------------------------------------------
                    BoundingBox = region_of_interest(x, yolo)
                    finalResult['carCount'] = len(BoundingBox)
                    q1Time = time.time() - startTime
                    stat1List.append(q1Time)
                    #---------------------------------------------------------------

                # # Query 2: To find the type of car (Hatchback or Sedan ? ) -----
                for car in BoundingBox:

                    if queryOption in [2,3]:
                        frameProcess +=1
                        processedFrameList.append(frameProcess)
                        # Call the classifier to test the ROI from the frames
                        carType = car_type_classifier(model,car[0])
                        print(f'\nThe car type is {carType}')

                        q2Time = time.time() - startTime
                        stat2List.append(q2Time)

                    #-------------------------------------------------------------- 

                    if queryOption in [3]:
                    # Query 3: To find the Color of the car----------------------   
                        carColor = colorDetector(car[0])

                        q3Time = time.time() - startTime
                        stat3List.append(q3Time)

                        print(f'The car color is {carColor}')
                        text = carColor + " " + carType
                        finalResult[carType][carColor] += 1
                #----------------------------------
                        top,left,bottom,right = car[1]
                        cv2.rectangle(x,(left,top-20),(right,bottom),(0,0,0),2)
                        cv2.putText(x, text, (left+10,top-10), cv2.FONT_HERSHEY_PLAIN,0.7, (0,255,0), 1)
                f_text = "car count: " + str(len(BoundingBox))
                print(f_text)
                cv2.putText(x, f_text, (50,50), cv2.FONT_HERSHEY_PLAIN,1, (0,255,0), 2)
                cv2.imshow("Processed_frames",x)
                cv2.waitKey(delay= 10)
                img_list.append(x)
                # Writing the frames into a video 
                out = cv2.VideoWriter('Waseem_Moiz_CaseStudyAssignment2_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (360,288))
                #----------------------------------
                #Adding the frame data to csv file:
                excelData = f"{frameNum}, {finalResult['Sedan']['Black']}, {finalResult['Sedan']['Silver']}, {finalResult['Sedan']['Red']}, {finalResult['Sedan']['White']}, {finalResult['Sedan']['Blue']}, {finalResult['Hatchback']['Black']}, {finalResult['Hatchback']['Silver']}, {finalResult['Sedan']['Red']}, {finalResult['Hatchback']['White']}, {finalResult['Hatchback']['Blue']},{finalResult['carCount']}"
                with open('ourResult.csv', 'a') as f:
                    f.write(excelData + '\n')

            frameNum += 1
            frameNumList.append(frameNum)

        #WAITING FOR FRAME IN QUEUE------------------------
        else: print('Waitiing for frame...')
        #--------------------------------------------------
        
        # Writing the frames into a video 
        for i in range(len(img_list)):
            out.write(img_list[i])
        out.release()


if __name__ == "__main__":
    car_type_model() # Calling car type model to train
    qu = mp.Queue()

    p1 = mp.Process(target=processFrame, args=(qu,))
    p2 = mp.Process(target=getFrame, args=(qu,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    #Calculating the F1 Score:
    statsGraph()
    calF1Score()
   
