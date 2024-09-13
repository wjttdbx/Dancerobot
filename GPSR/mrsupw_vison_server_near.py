#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import argparse
import sys
import time
import pyrealsense2 as rs
import cv2
from pyk4a import PyK4A, ColorResolution,Config
# import rospy
# from geometry_msgs.msg import *
#
# from std_msgs.msg import *
# from xm_msgs.msg import *
# from xm_msgs.srv import *

sys.path.append("/home/xm/xm_vision/darknet")
import darknet

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


def deal_with_language(s):
    global class_names
    lis = s.strip('[').strip(']').split(',')
    for word in lis:
        if word.strip().strip('\'') in types:
            if word.strip().strip('\'') =="biscuit":
                word='cookies'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="chip":
                word='potato wish'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="lays":
                word='potato chips'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="bread":
                word='bread'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="cookie":
                word='biscuit'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="handwash":
                word='Liquid soap'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="dishsoap":
                word='detergent'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="water":
                word='spring'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="cola":
                word='coke'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="orange juice":
                word='orange water'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="shampoo":
                word='shampoo'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="sprite":
                word='sprite'
                print('word is ', word)
                return word.strip().strip('\'')



def image_detection_original_no_image(image, network, class_names, thresh, fx, fy):
    global width, height
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


def deal_with_request(wobjects):
    print(wobjects)
    print(123)
    global is_debug
    is_found = False
    try_count = 0
    objects = []
    while not is_found:
        start_time = time.time()
        capture =  frame.get_color_frame()
        detections = image_detection_original_no_image(capture.color, network, class_names,
                                                       0.7,
                                                       width / capture.color.shape[1],
                                                       height / capture.color.shape[0])
        print(detections)
        try_count += 1
        if is_debug:
            drawn_image = darknet.draw_boxes(detections, capture.color, class_colors)
            cv2.putText(drawn_image, 'FPS: {:.2f}'.format((1 / (time.time() - start_time))), (20, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            cv2.putText(drawn_image, 'Press \'Q\' to Exit!'.format((1 / (time.time() - start_time))), (20, 70),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
        # 默认是0 如果没有识别到物体 那么返回的就是[0.0,0.0,0.0]
        coordinate = [0.0, 0.0, 0.0]
        for detection in detections:
            locals()
            x, y = int(detection[2][0]), int(detection[2][1])
            obj_coordinate = capture.transformed_depth_point_cloud[y][x]
            if is_debug:
                cv2.putText(drawn_image, '{}'.format(obj_coordinate), (x - 10, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 0, 0), thickness=2)
            print(objects)
            if detection[0].strip() in wobjects:
                print(wobjects)
                print(1111)
                # 说明找到了
                coordinate = capture.transformed_depth_point_cloud[y][x]
                if coordinate[0] or coordinate[1] or coordinate[2]:
                    # 单位转换成 m
                    coordinate = [coordinate[0] / 1000, coordinate[1] / 1000, coordinate[2] / 1000]
                    # 说明坐标非空！ 是有效坐标！
                    detect_result=detection[0].strip()
                    if detect_result.strip().strip('\'') == "cookies":
                        detect_result = 'biscuit'
                        print('word is ', detect_result)

                    if detect_result.strip().strip('\'') == "potato wish":
                        detect_result = 'chip'
                        print('word is ', detect_result)

                    if detect_result.strip().strip('\'') == "potato chips":
                        detect_result = 'lays'
                        print('word is ', detect_result)

                    if detect_result.strip().strip('\'') == "biscuit":
                        detect_result = 'cookie'
                        print('word is ', detect_result)

                    if detect_result.strip().strip('\'') == "Liquid soap":
                        detect_result = 'handwash'
                        print('word is ', detect_result)

                    if detect_result.strip().strip('\'') == "detergent":
                        detect_result = 'dishsoap'
                        print('word is ', detect_result)
                    if detect_result.strip().strip('\'') == "spring":
                        detect_result = 'water'
                        print('word is ', detect_result)

                    if detect_result.strip().strip('\'') == "coke":
                        detect_result = 'cola'
                        print('word is ', detect_result)
                    if detect_result.strip().strip('\'') == "orange water":
                        detect_result = 'orange juice'
                        print('word is ', detect_result)
                    if detect_result.strip().strip('\'') == "bread":
                        detect_result = 'bread'
                        print('word is ', detect_result)
                    if detect_result.strip().strip('\'') == "sprite":
                        detect_result = 'sprite'
                        print('word is ', detect_result)
                    if detect_result.strip().strip('\'') == "shampoo":
                        detect_result = 'shampoo'
                        print('word is ', detect_result)

                    is_found = True
        # print(colored('{}\'s coordinate is {}'.format(obj_name, coordinate), 'bright green'))
        if try_count >= 100:
            # 说明检测了100帧也没有找到想要的物体，视为检测失败
            print(colored("Detect FAIL!!", 'bright red'))
            return [0, 0, 0]
        if is_debug:
            cv2.imshow("Detected Image", drawn_image)
            if cv2.waitKey(1) in [ord("q"), ord("Q")]:
                cv2.destroyAllWindows()
                exit(0)
        if is_found:
            print("before return", coordinate)
            cv2.destroyAllWindows()
            return coordinate,detect_result

#
# def call_back(req):  # req的数据类型是xm_ObjectDetect()
#     res = xm_3ObjectDetectResponse()
#     objects = []
#     objects.append(deal_with_language(req.object_name1))
#     objects.append(deal_with_language(req.object_name2))
#     objects.append(deal_with_language(req.object_name3))
#     # 数据类型相当于xm_ObjectDetect
#     targetCor,object_result = deal_with_request(objects)
#     # 在此处添加
#     # 参考Object_detect.py
#     res.pos.point.x = targetCor[2]
#     res.pos.point.y = -targetCor[0]
#     res.pos.point.z = -targetCor[1]
#     res.object_name=object_result
#     res.pos.header.frame_id = "kinect2_rgb_link"
#     # res.pos.header.frame_id = "camera_body"
#     # res.pos.header.frame_id = "camera_visor"
#     res.pos.header.stamp = rospy.Time(0)
#     return res
#

if __name__ == "__main__":
    # True代表调试模式 False代表非调试模式
    parser = argparse.ArgumentParser()
    # True代表调试模式 False代表非调试模式
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    is_debug = args.debug
    types = ['biscuit', 'chip', 'lays', 'bread', 'cookie', 'handwash', 'dishsoap', 'water', 'sprite',
             'cola', 'orange juice', 'shampoo']
    type_classes = {
        'food': ['cookie', 'gum'],
        'drinks': ['water', 'milk', 'cola', 'tea_pi', 'spring', 'juice'],
        'toiletries': ['soap', 'lotion', 'toothpaste']
    }

    config_file = r'/home/xm/xm_vision/darknet/workspaces/xmTest_shopping/xm_vision.cfg'
    data_file = r'/home/xm/xm_vision/darknet/workspaces/xmTest_shopping/obj.data'
    weights = r'/home/xm/xm_vision/darknet/workspaces/xmTest_shopping/backup/yolov4-tiny-new_final.weights'
    batch_size = 1
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    # config = Config(color_resolution=ColorResolution.RES_1080P)
    # camera = PyK4A(config)
    # camera.start()
    rsPipeline = rs.pipeline()
    rsConfig = rs.config()
    rsConfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rsConfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rsPipeline.start(rsConfig)
    frame = rsPipeline.wait_for_frames()

    first_capture = frame.get_color_frame()
    fx = width / first_capture.get_data()[0]
    fy = height / first_capture.get_data()[1]

    objects = []
    objects.append(deal_with_language('cola'))
    objects.append(deal_with_language('sprite'))
    objects.append(deal_with_language('chip'))
    print(objects)
    # 数据类型相当于xm_ObjectDetect
    targetCor, object_result = deal_with_request(objects)
    print(targetCor)

    # rospy.init_node('object_detect')
    # service = rospy.Service('get_3position', xm_3ObjectDetect, call_back)
    # rospy.loginfo('object_detect')
    # print("time out")
    # rospy.spin()
