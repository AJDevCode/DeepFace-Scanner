#import statements, please have these installed to run the program
import cv2
from deepface import DeepFace

import os
import pandas as pd


# Testing, can run these commands if you want to test your single photo.
#img = cv2.imread("faces_pic/man1.jpg")
#results = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion"))


# Code to run multiple photos in directory,This will list the name of file, age, gender, race and emotion.

# Data to be collected from image
data = {
    "Name": [],
    "Age": [],
    "Gender": [],
    "Race": [],
    "Emotion": []

}

# Main Code

for file in os.listdir("faces_pic"):
    result = DeepFace.analyze(cv2.imread(f"faces_pic/{file}"), actions=("age", "gender", "race", "emotion"))
    data["Name"].append(file.split(".")[0])
    data["Age"].append(result[0]["age"])
    data["Gender"].append(result[0]["dominant_gender"])
    data["Race"].append(result[0]["dominant_race"])
    data["Emotion"].append(result[0]["dominant_emotion"])
FaceDetect = pd.DataFrame(data)
print(FaceDetect)
FaceDetect.to_csv("DescribePhotos.csv")

# CSV file outputs information to user in terminal.