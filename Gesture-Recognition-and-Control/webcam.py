import cv2
import numpy as np
import time
import os
import errno

import torch
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Normalize
from collections import OrderedDict, deque

from model import ConvColumn
import torch.nn as nn
import json

import imutils
from imutils.video import VideoStream, FileVideoStream, WebcamVideoStream, FPS
import argparse
import pyautogui
import configparser

qsize = 20  # size of queue to retain for 3D conv input
sqsize = 10 # size of queue for prediction stabilisation
num_classes = 9
threshold = 0.7


# from train_data.classes_dict in train.py
gesture_dict = {
    'Doing other things': 0, 0: 'Doing other things',
    'No gesture': 1, 1: 'No gesture', 
    'Stop Sign': 2, 2: 'Stop Sign', 
    'Swiping Down': 3, 3: 'Swiping Down', 
    'Swiping Left': 4, 4: 'Swiping Left', 
    'Swiping Right': 5, 5: 'Swiping Right', 
    'Swiping Up': 6, 6: 'Swiping Up', 
    'Turning Hand Clockwise': 7, 7: 'Turning Hand Clockwise', 
    'Turning Hand Counterclockwise': 8, 8: 'Turning Hand Counterclockwise'
}

# construct the argument parse and parse the arguments
str2bool = lambda x: (str(x).lower() == 'true')
parser = argparse.ArgumentParser()
# parser.add_argument('model')nppnpp
parser.add_argument("-e", "--execute", type=str2bool, default=True, help="Bool indicating whether to map output to keyboard/mouse commands")
parser.add_argument("-d", "--debug", type=str2bool, default=True, help="In debug mode, show webcam input")
parser.add_argument("-u", "--use_gpu", type=str2bool, default=True, help="Bool indicating whether to use GPU. False - CPU, True - GPU")
parser.add_argument("-g", "--gpus", default=[0], help="GPU ids to use")
# parser.add_argument("-c", "--config", default='./config.json', help="path to configuration file")
# parser.add_argument("-v", "--video", default='./gesture.mp4', help="Path to video file if using an offline file")
parser.add_argument("-v", "--video", default='', help="Path to video file if using an offline file")
parser.add_argument("-vb", "--verbose", default=2, help="Verbosity mode. 0- Silent. 1- Print info messages. 2- Print info and debug messages")
parser.add_argument("-cp", "--checkpoint", default="./model_best.pth.tar", help="Location of model checkpoint file")
parser.add_argument("-m", "--mapping", default="./mapping.ini", help="Location of mapping file for gestures to commands")
args = parser.parse_args()

parser.print_help()
# sys.exit(1)

print('Using %s for inference' % ('GPU' if args.use_gpu else 'CPU'))

# initialise some variables
verbose = args.verbose
device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

transform = Compose([
        ToPILImage(),
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

model = ConvColumn(num_classes)

# read in configuration file for mapping of gestures to keyboard keys
mapping = configparser.ConfigParser()
action = {}
if os.path.isfile(args.mapping):
    mapping.read(args.mapping)

    for m in mapping['MAPPING']: 
        val = mapping['MAPPING'][m].split(',')
        action[m] = {'fn': val[0], 'keys': val[1:]}  # fn: hotkey, press, typewrite

else:
    # print('[ERROR] Mapping file for gestures to keyboard keys is not found at ' + args.mapping)
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.mapping)


if args.use_gpu:
    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=args.gpus).to(device)

if os.path.isfile(args.checkpoint):
    # if (verbose>0): print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    new_state_dict = OrderedDict()

    for k, v in checkpoint.items():
        if(k == 'state_dict'):
            del checkpoint['state_dict']
            for j, val in v.items():
                name = j[7:] # remove `module.`
                new_state_dict[name] = val
            checkpoint['state_dict'] = new_state_dict
            break
    # start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    if (verbose>0): print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.checkpoint, checkpoint['epoch']))
else:
    # print("[ERROR] No checkpoint found at '{}'".format(args.checkpoint))
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.checkpoint)


# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
if verbose>0: print("[INFO] Attemping to start video stream...")

if (args.video == ''):
    vs = VideoStream(0, usePiCamera=False).start()
else:
    vs = FileVideoStream(args.video).start()

time.sleep(2.0)
fps = FPS().start()
Q = deque(maxlen=qsize)
SQ = deque(maxlen=sqsize)
act = deque(['No gesture', "No gesture"], maxlen=3)

# get first frame and use it to initialize our deque
frame = vs.read()
if frame is None:
    print('[ERROR] No video stream is available')

else:
    # frame = transform(frame)
    for i in range(qsize):
        Q.append(frame)
    if (verbose > 0): print('[INFO] Video stream started...')


# loop over the frames from the video stream
while(True):
    # grab the frame from the threaded video stream 
    frame = vs.read()
    if frame is None: 
        print('[ERROR] No video stream is available')
        break

    oframe = cv2.flip(frame.copy(),1)  # copy original frame for display later as mirror image
    
    # resize it to have a maximum height of 100 pixels (to be consistent with jester v1 dataset)
    frame = imutils.resize(frame, height=100)
    # (h, w) = frame.shape[:2]

    # frame = transform(frame)  # preprocessing function 
    Q.append(frame)

    # format data to torch
    imgs = []
    for img in Q:
        img = transform(img)
        imgs.append(torch.unsqueeze(img, 0))

    data = torch.cat(imgs)
    data = data.permute(1, 0, 2, 3)
    data = data[None, :, :, :, :]
    target = [2]
    target = torch.tensor(target)
    data = data.to(device)

    model.eval()  # set model to eval mode
    output = model(data)

    # send to softmax layer
    output = torch.nn.functional.softmax(output, dim=1)

    k = 5
    ts, pred = output.detach().cpu().topk(k, 1, True, True)
    top5 = [gesture_dict[pred[0][i].item()] for i in range(k)]

    pi = [pred[0][i].item() for i in range(k)]
    ps = [ts[0][i].item() for i in range(k)]
    top1 = top5[0] if ps[0] > threshold else gesture_dict[0]

    hist = {}
    for i in range(num_classes):
        hist[i] = 0
    for i in range(len(pi)):
        hist[pi[i]] = ps[i]
    SQ.append(list(hist.values()))

    ave_pred = np.array(SQ).mean(axis=0)
    top1 = gesture_dict[np.argmax(ave_pred)] if max(ave_pred) > threshold else gesture_dict[0]

    # show the output frame
    if (args.debug):
        cv2.putText(oframe, top1 + ' %.2f' % ps[0], (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(oframe, top1 + ' %.2f' % ps[0], (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.imshow("Frame", oframe)

    top1 = top1.lower()
    act.append(top1)

    # control an application based on mapped outputs
    # same top1 for consecutive frames
    if (act[0] != act[1] and len(set(list(act)[1:])) == 1):
        if top1 in action.keys():            
            t = action[top1]['fn']
            k = action[top1]['keys']
            
            if verbose > 1: print('[DEBUG]', top1, '-- ', t, str(k))
            if t == 'typewrite':
                pyautogui.typewrite(k)
            elif t == 'press':
                pyautogui.press(k)
            elif t == 'hotkey':
                for key in k:
                    pyautogui.keyDown(key)
                for key in k[::-1]:
                    pyautogui.keyUp(key)
                # pyautogui.hotkey(",".join(k))
    
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
    