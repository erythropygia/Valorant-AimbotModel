import cv2
import numpy as np
import mss

import ctypes
from ctypes import wintypes

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

# input type
class Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD),
                ("mi", ctypes.c_ulong)]

# mouse input 
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG)]

# SendInput
def send_input(inputs):
    nInputs = len(inputs)
    size_input = ctypes.sizeof(Input)
    arr_inputs = Input * nInputs
    pInputs = arr_inputs(*inputs)
    cb_size = ctypes.c_int(ctypes.sizeof(pInputs))
    ctypes.windll.user32.SendInput(nInputs, ctypes.byref(pInputs), cb_size)


def move_mouse(dx, dy):
    inputs = []
    mi = MouseInput(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.cast(None, wintypes.LPVOID))
    inputs.append(Input(INPUT_MOUSE, mi))
    send_input(inputs)
    click_left()


def click_left():
    inputs = []
    mi_down = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None)
    mi_up = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None)
    inputs.append(Input(INPUT_MOUSE, mi_down))
    inputs.append(Input(INPUT_MOUSE, mi_up))
    send_input(inputs)



# YOLOv4 model implementation
net = cv2.dnn.readNet("yolov4-tiny-custom_best.weights", "yolov4-tiny-custom.cfg")

# Class list (enemy_head, enemy)
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Output layer
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = []

# Screen resolution
mon = {"top": 0, "left": 0, "width": 1920, "height": 1080}


with mss.mss() as sct:
    while True:
        # Screenshot
        img = cv2.cvtColor(np.array(sct.grab(mon))[:, :, :3], cv2.COLOR_BGR2RGB)

        # YOLOv4 object detection
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Objects coordinates and label lists
        class_ids = []
        confidences = []
        boxes = []

        # Output layers process
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object coordinate calculate
                    center_x = int(detection[0] * img.shape[1])
                    center_y = int(detection[1] * img.shape[0])
                    w = int(detection[2] * img.shape[1])
                    h = int(detection[3] * img.shape[0])

                    # Boxes coordinate 
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maximum suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Drawing square for found object
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))  
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                print("X:" + str(x) + " Y:" + str(y) + " X+W:" + str(x + w) + " Y+H:" + str(y + h) + " Label:" + str(label))
                move_mouse(x,y)

