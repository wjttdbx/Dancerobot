from pyk4a import PyK4A, ColorResolution,Config
import cv2
import os

config = Config(color_resolution=ColorResolution.RES_1080P)
camera = PyK4A(config)
camera.start()

x = len(os.listdir('images'))
while True:
    capture = camera.get_capture()
    frame = capture.color
    cv2.imshow('frame' , frame)
    c = cv2.waitKey(10)
    if c == 27:
        break
    elif c == ord('s'):
        cv2.imwrite("images/{:0>4}.jpg".format(x) , frame)
        print("images/{:0>4}.jpg".format(x))
        x += 1
