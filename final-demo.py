from kmeans import *
# import required libraries
import struct
import sys
import serial
import binascii
import time
import numpy as np
import math

import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from IPython.display import HTML

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Local File Imports
from parse_bin_output import *


def getPointCloud(binDirPath):
    output_dict = parse_ADC(binDirPath)
    pointCloud = []
    for i in range(len(output_dict)):
        if 'pointCloud' in output_dict[i]:
            pointCloud.append(output_dict[i]['pointCloud'][:,0:3])
    return pointCloud

def extractCentroid(output_dict):
    clf = Kmeans(k=1)
    X_centroids = []
    for i in range(len(output_dict)):
        if 'pointCloud' in output_dict[i]:
            pointCloud = output_dict[i]['pointCloud'][:,0:3]
            _, centroids = clf.predict(pointCloud)
            X_centroids.append(centroids[0])
    return X_centroids

def plotCloud(pointCloud, centroids, predictions):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    def animate(i):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-3,3)
        ax.set_ylim(0,4)
        ax.set_zlim(-2,2)
        x,y,z = centroids[i]

        xyz = pointCloud[i]

        # xyz coordinate of radar
        radar_x = [0]
        radar_y = [0]
        radar_z = [0]

        ax.scatter(radar_x, radar_y, radar_z, c="b", marker='+', label='radar', s=[50])
        ax.scatter(x, y, z, c='black', s=25)
        ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c='g', s=15)
        ax.set_title(predictions[i])
    
    ani = animation.FuncAnimation(fig, animate, frames=len(centroids))
    ani.save('output.mp4', writer=animation.FFMpegWriter(fps=3))

binDirPath_jump = "binData/12_12_2023_15_16_45/"
binDirPath_stand = "binData/12_12_2023_15_19_20/"
output_dict_stand = parse_ADC(binDirPath_stand)
output_dict_jump = parse_ADC(binDirPath_jump)
X_centroids = []
y = []
clf = Kmeans(k=1)

for i in range(len(output_dict_stand)):
    if 'pointCloud' in output_dict_stand[i]:
        xyz = output_dict_stand[i]['pointCloud'][:,0:3]
        _, centroids = clf.predict(xyz)
        X_centroids.append(centroids[0])
        y.append("stand")

for i in range(len(output_dict_jump)):
    if 'pointCloud' in output_dict_jump[i]:
        xyz = output_dict_jump[i]['pointCloud'][:,0:3]
        _, centroids = clf.predict(xyz)
        X_centroids.append(centroids[0])
        y.append("jump")

# Assuming X_centroids contains centroid coordinates and y contains labels
X_train, X_test, y_train, y_test = train_test_split(X_centroids, y, test_size=0.2, random_state=42)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

binDirPath_test = "binData/12_12_2023_15_12_30/"
output_dict_test = parse_ADC(binDirPath_test)
X_centroids_test = extractCentroid(output_dict_test)

# Evaluate the model
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Make predictions on the test set
predictions = clf.predict(X_centroids_test)
pointCloud_test = getPointCloud(binDirPath_test)
plotCloud(pointCloud_test, X_centroids_test, predictions)
