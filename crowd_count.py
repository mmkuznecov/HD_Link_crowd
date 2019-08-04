import cv2
import numpy as np
import datetime, time
import os
import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch

from torchvision import datasets, transforms
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

model = CSRNet()

checkpoint = torch.load('0model_best.pth.tar',map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

from matplotlib import cm as c

print('Model loaded, start...')

cam = cv2.VideoCapture("udp://192.168.1.1:5600",cv2.CAP_FFMPEG)

ret,frame = cam.read()

cv2.imwrite('snaps/' + str(datetime.datetime.now()) + '.png', frame)
timeCheck = time.time()
future = 10 #interrupt in secs

while ret:
    if time.time() >= timeCheck:
        ret,frame = cam.read()
        timeCheck = time.time()+future

        path = 'snaps/' + str(datetime.datetime.now()) + '.png'
        cv2.imwrite(path, frame)

        img = transform(frame)
    
        output = model(img.unsqueeze(0))
        print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
        temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
        plt.imshow(temp,cmap = c.jet)
        plt.show()
        plt.imshow(plt.imread(path))
        plt.show()


    else:
        ret = cam.grab()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break