# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 00:48:14 2020

@author: Kieran Agarwal
implementation comes from: https://medium.com/@vivek8981/dicom-to-jpg-and-extract-all-patients-information-using-python-5e6dd1f1a07d
"""

import pydicom as dicom
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import csv
# make it True if you want in PNG format
PNG = False
# Specify the .dcm folder path
folder_path = "HeadDicoms"
# Specify the .jpg/.png folder path
jpg_folder_path = "headJPEGs"
images_path = os.listdir(folder_path)
# list of attributes available in dicom image
# download this file from the given link # https://github.com/vivek8981/DICOM-to-JPG
dicom_image_description = pd.read_csv("dicom_image_description.csv")

with open('Patient_Detail_head.csv', 'w', newline ='') as csvfile:
    fieldnames = list(dicom_image_description["Description"])
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(fieldnames)
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(folder_path, image))
        rows = []
        pixel_array_numpy = ds.pixel_array
        if PNG == False:
            image = image.replace('.dcm', '.jpg')
        else:
            image = image.replace('.dcm', '.png')
        cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
        if n % 50 == 0:
            print('{} image converted'.format(n))
        for field in fieldnames:
            if ds.data_element(field) is None:
                rows.append('')
            else:
                x = str(ds.data_element(field)).replace("'", "")
                y = x.find(":")
                x = x[y+2:]
                rows.append(x)
        writer.writerow(rows)