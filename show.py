import cv2 
import pickle
import matplotlib.pyplot as plt

#调整图像Size
def imageResizeTrain(image):
    maxD = 1024
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image
def imageResizeTest(image):
    maxD = 1024
    height,width,channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

#同上
# def imageResizeTest(image):
#     maxD = 1024
#     height,width,channel = image.shape
#     aspectRatio = width/height
#     if aspectRatio < 1:
#         newSize = (int(maxD*aspectRatio),maxD)
#     else:
#         newSize = (maxD,int(maxD/aspectRatio))
#     image = cv2.resize(image,newSize)
#     return image
def computeSIFT(image):
    return sift.detectAndCompute(image, None)
# def fetchKeypointFromFile(i):
#     filepath = "data/keypoints/" + str(imageList[i].split('.')[0]) + ".txt"
#     keypoint = []
#     file = open(filepath,'rb')
#     deserializedKeypoints = pickle.load(file)
#     file.close()
#     for point in deserializedKeypoints:
#         temp = cv2.KeyPoint(
#             x=point[0][0],
#             y=point[0][1],
#             size=point[1],
#             angle=point[2],
#             response=point[3],
#             octave=point[4],
#             class_id=point[5]
#         )
#         keypoint.append(temp)
#     return keypoint



# def fetchDescriptorFromFile(i):
#     filepath = "data/descriptors/" + str(imageList[i].split('.')[0]) + ".txt"
#     file = open(filepath,'rb')
#     descriptor = pickle.load(file)
#     file.close()
#     return descriptor
#匹配i和j
def calculateResultsFor(i,j,imageList,bf,imagesTxT):
# def calculateResultsFor(i,j,imageList,bf):
    def fetchKeypointFromFile(i):
        filepath = "data/keypoints/" + str(imageList[i].split('.')[0]) + ".txt"
        keypoint = []
        file = open(filepath, 'rb')
        deserializedKeypoints = pickle.load(file)
        file.close()
        for point in deserializedKeypoints:
            temp = cv2.KeyPoint(
                x=point[0][0],
                y=point[0][1],
                size=point[1],
                angle=point[2],
                response=point[3],
                octave=point[4],
                class_id=point[5]
            )
            keypoint.append(temp)
        return keypoint

    def fetchDescriptorFromFile(i):
        filepath = "data/descriptors/" + str(imageList[i].split('.')[0]) + ".txt"
        file = open(filepath, 'rb')
        descriptor = pickle.load(file)
        file.close()
        return descriptor

    def calculateMatches(des1, des2):
        matches = bf.knnMatch(des1, des2, k=2)
        topResults1 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                topResults1.append([m])

        matches = bf.knnMatch(des2, des1, k=2)
        topResults2 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                topResults2.append([m])

        topResults = []
        for match1 in topResults1:
            match1QueryIndex = match1[0].queryIdx
            match1TrainIndex = match1[0].trainIdx

            for match2 in topResults2:
                match2QueryIndex = match2[0].queryIdx
                match2TrainIndex = match2[0].trainIdx

                if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                    topResults.append(match1)
        return topResults
    def getPlot(image1, image2, keypoint1, keypoint2, matches):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        matchPlot = cv2.drawMatchesKnn(
            image1,
            keypoint1,
            image2,
            keypoint2,
            matches,
            None,
            # 线的颜色
            [100, 100, 100],
            flags=2
        )
        return matchPlot
    # def getPlotFor(i, j, keypoint1, keypoint2, matches):
    #     image1 = imageResizeTest(cv2.imread("data/images/" + imageList[i]))
    #     image2 = imageResizeTest(cv2.imread("data/images/" + imageList[j]))
    #     return getPlot(image1, image2, keypoint1, keypoint2, matches)
    def getPlotFor(i, j, keypoint1, keypoint2, matches):
        image1 = imagesTxT[0]
        image2 = imagesTxT[1]
        return getPlot(image1, image2, keypoint1, keypoint2, matches)
    def calculateScore(matches, keypoint1, keypoint2):
        return 100 * (matches / min(keypoint1, keypoint2))
    keypoint1 = fetchKeypointFromFile(i)
    descriptor1 = fetchDescriptorFromFile(i)
    keypoint2 = fetchKeypointFromFile(j)
    descriptor2 = fetchDescriptorFromFile(j)
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    plot = getPlotFor(i,j,keypoint1,keypoint2,matches)
    # print(len(matches),len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))
    # print(score)
    # plt.imshow(plot),plt.show()
    return plot
# def getPlotFor(i,j,keypoint1,keypoint2,matches):
#     image1 = imageResizeTest(cv2.imread("data/images/" + imageList[i]))
#     image2 = imageResizeTest(cv2.imread("data/images/" + imageList[j]))
#     return getPlot(image1,image2,keypoint1,keypoint2,matches)
# def calculateScore(matches,keypoint1,keypoint2):
#     return 100 * (matches/min(keypoint1,keypoint2))
# def calculateMatches(des1, des2):
#     matches = bf.knnMatch(des1, des2, k=2)
#     topResults1 = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             topResults1.append([m])
#
#     matches = bf.knnMatch(des2, des1, k=2)
#     topResults2 = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             topResults2.append([m])
#
#     topResults = []
#     for match1 in topResults1:
#         match1QueryIndex = match1[0].queryIdx
#         match1TrainIndex = match1[0].trainIdx
#
#         for match2 in topResults2:
#             match2QueryIndex = match2[0].queryIdx
#             match2TrainIndex = match2[0].trainIdx
#
#             if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
#                 topResults.append(match1)
#     return topResults
# def getPlot(image1,image2,keypoint1,keypoint2,matches):
#     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#     image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#     matchPlot = cv2.drawMatchesKnn(
#         image1,
#         keypoint1,
#         image2,
#         keypoint2,
#         matches,
#         None,
#         #线的颜色
#         [100,100,100],
#         flags=2
#     )
#     return matchPlot

if __name__ == '__main__':
    imageList = ["taj1.jpg","taj2.jpg","eiffel1.jpg","eiffel2.jpg","liberty1.jpg","liberty2.jpg","robert1.jpg","tom1.jpg","ironman1.jpg","ironman2.jpg","ironman3.jpg","darkknight1.jpg","darkknight2.jpg","book1.jpg","book2.jpg","img1.jpg","img2.jpg","1.jpg",'2.jpg']
    imagesBW = []
    for imageName in imageList:
        imagePath = "data/images/" + str(imageName)
        #将图像转换成1024size
        imagesBW.append(imageResizeTrain(cv2.imread(imagePath,0))) # flag 0 means grayscale
        sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for i,image in enumerate(imagesBW):
        print("Starting for image: " + imageList[i])
        #计算SIFT,返回KeypointTemp,descriptor
        keypointTemp, descriptorTemp = computeSIFT(image)
        keypoints.append(keypointTemp)
        descriptors.append(descriptorTemp)
        print("  Ending for image: " + imageList[i])
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
    calculateResultsFor(17,18,imageList,bf)