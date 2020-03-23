from ctypes import *
import math
import random
import cv2
import numpy as np
import pyglet
import time
from random import randint
import pygame


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL(b"/home/melih/Desktop/yolo/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = nparray_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

global play_time
play_time = time.time()

pygame.mixer.init()

def play_audio():
    global play_time
    i = randint(1, 5)
    name = str(i) + ".mp3"
    #music = pyglet.resource.media('hand_sound/' + name)
    #music.play()
    pygame.mixer.music.load('hand_sound/' + name)
    pygame.mixer.music.play()
    play_time = time.time()

def overlap(l1, r1, l2, r2):
    Rect1_x1 = l1.x
    Rect1_y1 = l1.y
    Rect1_x2 = r1.x
    Rect1_y2 = r1.y

    Rect2_x1 = l2.x
    Rect2_y1 = l2.y
    Rect2_x2 = r2.x
    Rect2_y2 = r2.y

    if (Rect2_x2 > Rect1_x1 and Rect2_x2 < Rect1_x2) or \
        (Rect2_x1 > Rect1_x1 and Rect2_x1 < Rect1_x2):
       x_match = True
    else:
        x_match = False
    if (Rect2_y2 > Rect1_y1 and Rect2_y2 < Rect1_y2) or \
        (Rect2_y1 > Rect1_y1 and Rect2_y1 < Rect1_y2):
        y_match = True
    else:
        y_match = False
    
    if x_match and y_match:
        return True
    else:
        return False

def draw_recs(image, objects, names):
    face_l = None
    face_r = None

    hand1_l = None
    hand1_r = None

    hand2_l = None
    hand2_r = None

    for i in range (objects.shape[0]):
        hand_counter = 0
        if (names[i].decode() == "face"):
            face_l = Point(int(objects[i][0]) - int(objects[i][2]/2), int(objects[i][1]) - int(objects[i][3]/2))
            face_r = Point(int(objects[i][0]) + int(objects[i][2]/2), int(objects[i][1]) + int(objects[i][3]/2))
        elif (names[i].decode() == "hand"):
            if hand_counter == 0:
                hand1_l = Point(int(objects[i][0]) - int(objects[i][2]/2), int(objects[i][1]) - int(objects[i][3]/2))
                hand1_r = Point(int(objects[i][0]) + int(objects[i][2]/2), int(objects[i][1]) + int(objects[i][3]/2))
                hand_counter = hand_counter + 1
            elif hand_counter == 1:
                hand2_l = Point(int(objects[i][0]) - int(objects[i][2]/2), int(objects[i][1]) - int(objects[i][3]/2))
                hand2_r = Point(int(objects[i][0]) + int(objects[i][2]/2), int(objects[i][1]) + int(objects[i][3]/2)) 

        rx = (int(objects[i][0]) - int(objects[i][2]/2), int(objects[i][1]) - int(objects[i][3]/2))
        ry = (int(objects[i][0]) + int(objects[i][2]/2), int(objects[i][1]) + int(objects[i][3]/2))
        tx = int(objects[i][0])
        ty = int(objects[i][1])
        cv2.putText(frame, names[i].decode(), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, 1)
        cv2.rectangle(frame, rx, ry, (0,0,255), 2)

        if (face_l != None and face_r != None):
            if (hand1_l != None and hand1_r != None):
                if(overlap(face_l, face_r, hand1_l, hand1_r)):
                    now = time.time()
                    if now - play_time > 2.5:
                        play_audio()
            if (hand2_l != None and hand2_r != None):
                if(overlap(face_l, face_r, hand2_l, hand2_r)): 
                    play_audio()

if __name__ == "__main__":
    net = load_net(b"/home/melih/Desktop/yolo/darknet/cfg/yolov3-voc_inf.cfg", b"/home/melih/Desktop/yolo/darknet/yolov3-voc.backup", 0)
    meta = load_meta(b"/home/melih/Desktop/yolo/darknet/cfg/obj.data")
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 960)
    
    if not (cap.isOpened()):
        print("Could not open video device")

    while (True):
        ret, frame = cap.read()
        ret, frame1 = cap.read()
        r = detect(net, meta, frame)
        if len(r) != 0: 
            npr = np.asarray(r)
            draw_recs(frame, npr[:,2], npr[:,0])
        cv2.imshow("frame", frame1)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
