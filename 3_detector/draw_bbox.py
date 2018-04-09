import os
import cv2
import sys
import csv
import numpy as np
import config as cf

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        print("Creating "+in_dir+"...")
        os.makedirs(in_dir)

if __name__ == "__main__":
    check_and_mkdir('./results/bbox/')

    for file_number in range(1, (27+1)):
        print("| Predicting Bounding Box for TEST%d..." %file_number)
        original_img = cv2.imread("/home/bumsoo/Data/test/MICCAI_img/TEST%d.png" %file_number)
        mask_img = cv2.imread('./results/masks/TEST%d.png' %file_number)

        ret, threshed_img = cv2.threshold(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(threshed_img, cv2.MORPH_CLOSE, kernel, iterations=4)

        _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0

        # Predictions (GREEN)
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if (area > 30**2):
                # ellipse
                #ellipse = cv2.fitEllipse(cnt)
                #cv2.ellipse(original_img, ellipse, (0,255,0), 2)

                # bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(original_img, (x,y), (x+w, y+h), (0,255,0), 2)

        # Ground truth (RED)
        with open(cf.test_dir+'%d/TEST%d.csv' %(file_number, file_number)) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x, y, w, h = map(int, row[1:])

                cv2.rectangle(original_img, (x,y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imwrite('./results/bbox/TEST%d.png' %file_number, original_img)
