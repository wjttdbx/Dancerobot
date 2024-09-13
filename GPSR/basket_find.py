#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import pyrealsense2 as rs
import numpy as np

import rospy
from geometry_msgs.msg import *

from std_msgs.msg import *
from xm_msgs.msg import *
from xm_msgs.srv import *



rsPipeline = rs.pipeline()
rsConfig = rs.config()
rsConfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rsConfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rsPipeline.start(rsConfig)

def deal_with_request():
    count=0
    sum_count=5
    x=0
    while count<sum_count:
        frames = rsPipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        HSVimg = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(HSVimg, (35, 43, 46), (77, 255, 255))
        # print(mask)
        cv2.imshow("frame", mask)
        depth_sum = 0
        sum = 0
        for i in range(640):
            for j in range(480):
                if mask[j][i] == 255:
                    if depth_image[j][i] <= 5000:
                        depth_sum += depth_image[j][i]
                        sum += 1
        if sum != 0:
            # print(depth_sum / sum)
            x+=depth_sum/sum/1000
            # print(x/sum)
        k = cv2.waitKey(10)
        if k & 0xff == ord('q') or k == 27:  # 按q或esc
            cv2.destoryAllWindows()
            break
        count+=1
    print(x/(count-1))
    return x/sum_count

def call_back(req):
    res= xm_ObjectDetectResponse()
    res.object.pos.point.x = deal_with_request()
    res.object.pos.point.y = 0
    res.object.pos.point.z = -0.15
    res.object.pos.header.frame_id = "kinect2_rgb_link"
    res.object.pos.header.stamp = rospy.Time(0)
    return res

if __name__ == "__main__":
    rospy.init_node('basket_detect')
    service = rospy.Service('get_position', xm_ObjectDetect, call_back)
    rospy.loginfo('basket_detect')
    rospy.spin()
    # deal_with_request()