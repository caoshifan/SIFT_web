# -*—coding : utf-8 -*-
# @Time : 2024/5/28 18:02
# @Author :
# @File : web
# @Project : SIFTImageSimilarity
from flask import Flask, request, jsonify, render_template
import base64
import io
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from show import imageResizeTrain,  calculateResultsFor

app = Flask(__name__)


@app.route('/deal_image', methods=['POST'])
def deal_image():
    images = request.files.getlist("images")
    imagesBW = []

    def computeSIFT(image):
        return sift.detectAndCompute(image, None)

    def imageResizeTest(image):
        maxD = 1024
        height, width, channel = image.shape
        aspectRatio = width / height
        if aspectRatio < 1:
            newSize = (int(maxD * aspectRatio), maxD)
        else:
            newSize = (maxD, int(maxD / aspectRatio))
        image = cv2.resize(image, newSize)
        return image

    i = 0
    imageList = []
    imagesTxT = []
    for image in images:
        imagelist = str(i) + '.jpg'
        i += 1
        imageList.append(imagelist)
        in_memory_file = io.BytesIO()
        image.save(in_memory_file)
        # data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 0
        img_array = cv2.imdecode(data, color_image_flag)
        img_arrayt = cv2.imdecode(data,1)
        imagesBW.append(imageResizeTrain(img_array))
        imagesTxT.append(imageResizeTest(img_arrayt))
        sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for i,image in enumerate(imagesBW):
        # print("Starting for image: " + imageList[i])
        #计算SIFT,返回KeypointTemp,descriptor
        keypointTemp, descriptorTemp = computeSIFT(image)
        keypoints.append(keypointTemp)
        descriptors.append(descriptorTemp)
        # print("  Ending for image: " + imageList[i])
    #保存两个文件
    for i,keypoint in enumerate(keypoints):
        deserializedKeypoints = []
        filepath = "data/keypoints/" + str(imageList[i].split('.')[0]) + ".txt"
        for point in keypoint:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
            deserializedKeypoints.append(temp)
        with open(filepath, 'wb') as fp:
            pickle.dump(deserializedKeypoints, fp)
    for i,descriptor in enumerate(descriptors):
        filepath = "data/descriptors/" + str(imageList[i].split('.')[0]) + ".txt"
        with open(filepath, 'wb') as fp:
            pickle.dump(descriptor, fp)
    bf = cv2.BFMatcher()
    matchPlot = calculateResultsFor(0,1,imageList,bf,imagesTxT)
    rgb_image = cv2.cvtColor(matchPlot, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', rgb_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': base64_image})



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
