import cv2
import matplotlib.pyplot as plt
import glob, os
from random import randint

def drawRects(path, im, color):
    with open(path) as f:
        rectFile = f.readlines()

    while len(rectFile) > 0:
        verticies = []
        for i in range(4):
            line = rectFile[i].rstrip("\n\r").split(" ")
            verticies.append((int(float(line[0])), int(float(line[1]))))

        
        cv2.line(im, verticies[0], verticies[1], color, 1)
        cv2.line(im, verticies[1], verticies[2], color, 2)
        cv2.line(im, verticies[2], verticies[3], color, 1)
        cv2.line(im, verticies[3], verticies[0], color, 2)

        rectFile = rectFile[4:]


imagePaths = glob.glob("./cornell/*.png")


for dispImages in range(5):
    randIndex = randint(0, len(imagePaths))
    imPath = imagePaths[randIndex]
    posPath = imPath[0:-5] + "cpos.txt"
    negPath = imPath[0:-5] + "cneg.txt"

    im = cv2.imread(imPath)

    drawRects(posPath, im, (100, 255, 100))
    #drawRects(negPath, im, (0, 0, 255))



    cv2.imshow('Test image',im)
    cv2.waitKey(0)


cv2.destroyAllWindows()