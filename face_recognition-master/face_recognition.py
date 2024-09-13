#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : face_recognition.py
#   Author      : YunYang1994
#   Created date: 2020-02-23 16:34:47
#   Description :
#
#================================================================
import argparse
import os
import cv2
import time
import numpy as np
import tensorflow as tf
from pyk4a import PyK4A, ColorResolution,Config
from models import MobileFaceNet
from mtcnn import pnet, rnet, onet
from utils import detect_face, align_face, recognize_face
# import rospy
# from geometry_msgs.msg import *
# from std_msgs.msg import *
# from xm_msgs.msg import *
# from xm_msgs.srv import *

codeCodes = {
    'black': '0;30', 'bright gray': '0;37',
    'blue': '0;34', 'white': '1;37',
    'green': '0;32', 'bright blue': '1;34',
    'cyan': '0;36', 'bright green': '1;32',
    'red': '0;31', 'bright cyan': '1;36',
    'purple': '0;35', 'bright red': '1;31',
    '***': '0;33', 'bright purple': '1;35',
    'grey': '1;30', 'bright yellow': '1;33',
}


def colored(text, color='green'):
    return "\033[" + codeCodes[color] + "m" + text + "\033[0m"

model = MobileFaceNet()
is_found=False
def deal_with_request():
    global is_debug
    is_found = False
    try_count = 0
    while not is_found:
        cap = camera.get_capture()
        image = cap.color
        # resize image
        image_h, image_w, _ = image.shape
        new_h, new_w = int(0.5*image_h), int(0.5*image_w)
        image = cv2.resize(image, (new_w, new_h))
        print(try_count)
        org_image = image.copy()
        # detecting faces
        # t1 = time.time()
        image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
        total_boxes, points = detect_face(image, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)
        print(888888)
        # t2 = time.time()
        # print("time: %.2fms" %((t2-t1)*1000))
        try_count += 1
        for idx, (bounding_box, keypoints) in enumerate(zip(total_boxes, points.T)):
            bounding_boxes = {
                    'box': [int(bounding_box[0]), int(bounding_box[1]),
                            int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])],
                    'confidence': bounding_box[-1],
                    'keypoints': {
                            'left_eye': (int(keypoints[0]), int(keypoints[5])),
                            'right_eye': (int(keypoints[1]), int(keypoints[6])),
                            'nose': (int(keypoints[2]), int(keypoints[7])),
                            'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                            'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                    }
                }

            bounding_box = bounding_boxes['box']
            keypoints = bounding_boxes['keypoints']

            cv2.circle(org_image,(keypoints['left_eye']),   2, (255,0,0), 3)
            cv2.circle(org_image,(keypoints['right_eye']),  2, (255,0,0), 3)
            cv2.circle(org_image,(keypoints['nose']),       2, (255,0,0), 3)
            cv2.circle(org_image,(keypoints['mouth_left']), 2, (255,0,0), 3)
            cv2.circle(org_image,(keypoints['mouth_right']),2, (255,0,0), 3)
            cv2.rectangle(org_image,
                        (bounding_box[0], bounding_box[1]),
                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (0,255,0), 2)
            # align face and extract it out
            align_image = align_face(image, keypoints)
            marigin = 16
            xmin = max(bounding_box[0] - marigin, 0)
            ymin = max(bounding_box[1] - marigin, 0)
            xmax = min(bounding_box[0] + bounding_box[2] + marigin, new_w)
            ymax = min(bounding_box[1] + bounding_box[3] + marigin, new_h)

            crop_image = align_image[ymin:ymax, xmin:xmax, :]
            if crop_image is not None:
                t1 = time.time()

                embedding = model(crop_image)
                person = recognize_face(embedding)
                print(person)
                if person ==None:
                    print(1111111111)
                if person!="Unknown":
                    is_found = True
                    break
                cv2.putText(org_image, person, (bounding_box[0], bounding_box[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                                                                    1., (0, 0, 255), 3, lineType=cv2.LINE_AA)
                t2 = time.time()
                print("time: %.2fms" %((t2-t1)*1000))


        if try_count >= 100:
            # 说明检测了100帧也没有找到想要的物体，视为检测失败
            print(colored("Detect FAIL!!", 'bright red'))
            print("None")
            cv2.destroyAllWindows()
            return "None"
        if is_debug:
            cv2.imshow("Detected Image", org_image)
            if cv2.waitKey(1) in [ord("q"), ord("Q")]:
                cv2.destroyAllWindows()
                exit(0)
        if is_found:
            print(person)
            cv2.destroyAllWindows()
            return str(person)
#
# def call_back(req):  # req的数据类型是xm_ObjectDetect()
#     res = xm_FaceDetectResponse()
#     # 数据类型相当于xm_ObjectDetect
#     res.people_name=deal_with_request()
#     print(res)
#     return res
#
#
# if __name__ == "__main__":
#     # True代表调试模式 False代表非调试模式
#     parser = argparse.ArgumentParser()
#     # True代表调试模式 False代表非调试模式
#     parser.add_argument('-d', '--debug', type=bool, default=False)
#     args = parser.parse_args()
#     is_debug = args.debug
#
#     config = Config(color_resolution=ColorResolution.RES_1080P)
#     camera = PyK4A(config)
#     camera.start()
#     # person=deal_with_request()
#     # print(person)
#
#     rospy.init_node('face_detect')
#     service = rospy.Service('get_name', xm_FaceDetect, call_back)
#     rospy.loginfo('face_detect')
#     # print("time out")
#     rospy.spin()
