#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time

import pyrealsense2 as rs
import numpy as np
import cv2
import darknet
from mrsupw_vison_server_near import deal_with_language ,colored
import rospy
from std_msgs.msg import *
from xm_msgs.msg import *
from xm_msgs.srv import *
from geometry_msgs.msg import *

rsConfig = None
rsPipeline = None
config_file = r'/home/xm/xm_vision/darknet/workspaces/xmTest_shopping/xm_vision.cfg'
data_file = r'/home/xm/xm_vision/darknet/workspaces/xmTest_shopping/obj.data'
weights = r'/home/xm/xm_vision/darknet/workspaces/xmTest_shopping/backup/yolov4-tiny-new_4000.weights'
batch_size = 1

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

is_debug = False
pc = None
points = None



def colored(text, color='green'):
    return "\033[" + codeCodes[color] + "m" + text + "\033[0m"


def deal_with_language(s):
    global class_names
    lis = s.strip('[').strip(']').split(',')
    for word in lis:
        print('word is ', word)
        if word.strip().strip('\'') in types:
            return word.strip().strip('\'')


def image_detection_original_no_image(image, network, class_names, thresh, fx, fy ,width ,height):
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    for i in range(len(detections)):
        detections[i] = list(detections[i])
        detections[i][2] = list(detections[i][2])
        detections[i][2][0] = (width / 2 / fx) - (1 / fx) * (width / 2 - detections[i][2][0]) if detections[i][2][
                                                                                                     0] <= width / 2 \
            else (1 / fx) * (detections[i][2][0] - width / 2) + (width / 2 / fx)
        detections[i][2][1] = (height / 2 / fy) - (1 / fy) * (height / 2 - detections[i][2][1]) if detections[i][2][
                                                                                                       1] <= height / 2 \
            else (1 / fy) * (detections[i][2][1] - height / 2) + (height / 2 / fy)
        detections[i][2][2] /= fx
        detections[i][2][3] /= fy
    darknet.free_image(darknet_image)
    return detections

def deal_with_request(obj_name):
    rsPipeline = rs.pipeline()
    rsConfig   = rs.config()
    rsConfig.enable_stream(rs.stream.color , 640 , 480 , rs.format.bgr8, 30)
    rsConfig.enable_stream(rs.stream.depth , 640 , 480 , rs.format.z16 , 30)
    rsPipeline.start(rsConfig)
    is_found = False
    try_count = 0
    objects = []
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    while not is_found:
        start_time = time.time()
        frames = rsPipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_data  = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        pc = rs.pointcloud()
        points = rs.points()
        cv2.imshow('realsense', color_image)
        cv2.waitKey(20)
        print(color_image.shape)

        detections = image_detection_original_no_image(color_image ,
                                                         network ,
                                                         class_names ,
                                                         0.7,
                                                         width  / color_image.shape[1],
                                                         height / color_image.shape[0],
                                                         width,
                                                         height
                                                        )
        print(detections)
        # print('qwertyuiop')
        try_count += 1
        if True:
            drawn_image = darknet.draw_boxes(detections, color_image, class_colors)
            cv2.imwrite('pic.jpg',drawn_image)
            cv2.putText(drawn_image, 'FPS: {:.2f}'.format((1 / (time.time() - start_time))), (20, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            cv2.putText(drawn_image, 'Press \'Q\' to Exit!'.format((1 / (time.time() - start_time))), (20, 70),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
        coordinate = [0.0 ,0.0 ,0.0]
        for detection in detections:
            locals()
            exit(0)
            x, y = int(detection[2][0]), int(detection[2][1])
            dis = depth_frame.get_distance(x , y)
            print(dis)
            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin , (x , y) ,dis)
            camera_xyz = np.round(np.array(camera_xyz) , 3)
            camera_xyz = camera_xyz.tolist()
            print(x, y)
            print(camera_xyz)
            # cv2.imshow('pic.jpg' , drawn_image)
            obj_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin , (x , y) ,dis)
            if is_debug:
                cv2.putText(drawn_image, '{}'.format(obj_coordinate), (x - 10, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 0, 0), thickness=2)
            if detection[0].strip() == obj_name:
                # 说明找到了
                coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin , (x , y) ,dis)
                if coordinate[0] or coordinate[1] or coordinate[2]:
                    # 单位转换成 m
                    coordinate = [coordinate[0] / 1000, coordinate[1] / 1000, coordinate[2] / 1000]
                    # 说明坐标非空！ 是有效坐标！

                    is_found = True
        print(colored('{}\'s coordinate is {}'.format(obj_name, coordinate), 'bright green'))

        # if try_count >= 100:
        #     rsPipeline.stop()
        #     cv2.imwrite('piccc.jpg' , color_image)
        #     # 说明检测了100帧也没有找到想要的物体，视为检测失败
        #     print(colored("Detect FAIL!!", 'bright red'))
        #     return [0, 0, 0]
        if True:
            # cv2.imshow("Detected Image", drawn_image)
            if cv2.waitKey(1) in [ord("q"), ord("Q")]:
                cv2.destroyAllWindows()
                exit(0)
        if is_found:
            print("before return", coordinate)
            cv2.destroyAllWindows()
            return coordinate
        end_time = time.time()
        frames = rsPipeline.wait_for_frames()

        # print("FPS is ", 1 / (end_time - start_time))
    rsPipeline.stop()
    print("stop")






def call_back(req):
    res = xm_ObjectDetectResponse()
    object_name = deal_with_language(req.object_name)
    print('object_name_rs is ' , object_name)
    target = deal_with_request(object_name)

    res.object.pos.point.x = target[2]
    res.object.pos.point.y = -target[0]
    res.object.pos.point.z = -target[1]
    res.object.pos.header.frame_id = "kinect2_rgb_link"
    # res.pos.header.frame_id = "camera_body"
    # res.pos.header.frame_id = "camera_visor"
    res.object.pos.header.stamp = rospy.Time(0)
    res.object.state = 1
    return res


if __name__ == "__main__":
    deal_with_request("666")
    # True代表调试模式 False代表非调试模式
    parser = argparse.ArgumentParser()
    # True代表调试模式 False代表非调试模式
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    is_debug = args.debug

    types = ['cookies','potato wish', 'potato chips','spring','orange water']
    type_classes = {
        'food': ['cookie', 'gum'],
        'drinks': ['water', 'milk', 'cola', 'tea_pi', 'spring', 'juice'],
        'toiletries': ['soap', 'lotion', 'toothpaste']
    }

    rsPipeline = rs.pipeline()
    rsConfig   = rs.config()
    rsConfig.enable_stream(rs.stream.color , 640 , 480 , rs.format.bgr8, 30)
    rsConfig.enable_stream(rs.stream.depth , 640 , 480 , rs.format.z16 , 30)
    rsPipeline.start(rsConfig)

    #
    rospy.init_node('object_detect_rs')
    service_rs = rospy.Service('get_position' , xm_ObjectDetect , call_back)
    rospy.loginfo('object_detect_rs')
    rospy.spin()
