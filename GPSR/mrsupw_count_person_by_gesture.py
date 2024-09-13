#!/usr/bin/python3
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]

import argparse
import threading
import rospy
import cv2
from aip import AipBodyAnalysis
from geometry_msgs.msg import *
from pyk4a import PyK4A, ColorResolution, Config
from xm_msgs.msg import *
from xm_msgs.srv import *
from std_msgs.msg import *

color_codes = {
    'black': '0;30', 'bright gray': '0;37',
    'blue': '0;34', 'white': '1;37',
    'green': '0;32', 'bright blue': '1;34',
    'cyan': '0;36', 'bright green': '1;32',
    'red': '0;31', 'bright cyan': '1;36',
    'purple': '0;35', 'bright red': '1;31',
    '***': '0;33', 'bright purple': '1;35',
    'grey': '1;30', 'bright yellow': '1;33',
}

clothes_color = {
    '红': 'red', '橙': 'orange',
    '黄': 'yellow', '绿': 'green',
    '蓝': 'blue', '紫': 'purple',
    '粉': 'pink', '黑': 'black',
    '白': 'white', '灰': 'grey', '棕': 'brown'
}

analysis_result = []
attr_result = []
gestures = ['waving', 'left_point', 'right_point', 'left_raise', 'right_raise',
            'standing', 'sitting', 'lying',
            'male', 'female', 'young male', 'young female', 'old male', 'old female', 'name']
POINT_X_THRESHOLD = 100
POINT_Y_THRESHOLD = 60
HANDS_UP_THRESHOLD = 50

STANDING_THRESHOLD = 125
SITTING_THRESHOLD = 100
LYING_THRESHOLD = 60


def colored(text, color='green'):
    return "\033[" + color_codes[color] + "m" + str(text) + "\033[0m"


def drawDebugImage(image, final_results):
    for final_result in final_results:
        location = final_result['location']
        cv2.rectangle(image, (round(location['left']), round(location['top'])),
                      (round(location['left'] + location['width']), round(location['top'] + location['height'])),
                      (0, 255, 0), thickness=2)
        cv2.circle(image, final_result['right_wrist'], 3, (255, 255, 0), thickness=3)
        cv2.circle(image, final_result['right_elbow'], 3, (255, 255, 0), thickness=3)
        cv2.circle(image, final_result['right_shoulder'], 3, (255, 255, 0), thickness=3)
        cv2.circle(image, final_result['right_hip'], 3, (255, 255, 0), thickness=3)
        cv2.circle(image, final_result['right_knee'], 3, (255, 255, 0), thickness=3)
        cv2.circle(image, final_result['right_ankle'], 3, (255, 255, 0), thickness=3)
        cv2.circle(image, final_result['left_wrist'], 3, (0, 255, 255), thickness=3)
        cv2.circle(image, final_result['left_elbow'], 3, (0, 255, 255), thickness=3)
        cv2.circle(image, final_result['left_shoulder'], 3, (0, 255, 255), thickness=3)
        cv2.circle(image, final_result['left_hip'], 3, (0, 255, 255), thickness=3)
        cv2.circle(image, final_result['left_knee'], 3, (0, 255, 255), thickness=3)
        cv2.circle(image, final_result['left_ankle'], 3, (0, 255, 255), thickness=3)
    return image


def deal_with_bodyAttr_result(img_bytes):
    global attr_result
    attr_result = client.bodyAttr(img_bytes)
    if attr_result["person_num"] > 0:
        result_data = []
        for person_info in attr_result['person_info']:
            person_data = {}
            attributes = person_info["attributes"]
            person_data["gender"] = attributes["gender"]["name"]
            person_data["age"] = attributes["age"]['name']
            person_data["upper_color"] = attributes["upper_color"]['name']
            person_data["glasses"] = attributes["glasses"]['name']
            person_data["face_mask"] = attributes["face_mask"]['name']
            person_data["headwear"] = attributes["face_mask"]['name']
            person_data["headwear"] = attributes["headwear"]['name']
            person_data["location"] = person_info["location"]
            person_data["center"] = (round(person_data["location"]["left"] + person_data["location"]["width"] / 2),
                                     round(person_data["location"]["top"] + person_data["location"]["height"] / 2))
            result_data.append(person_data)
        attr_result = result_data
    else:
        attr_result = []


def deal_with_bodyAnalysis_result(img_bytes):
    global analysis_result
    analysis_result = client.bodyAnalysis(img_bytes)
    if analysis_result["person_num"] > 0:
        result_data = []
        for person_info in analysis_result['person_info']:
            person_data = {}
            body_parts = person_info["body_parts"]
            person_data["right_wrist"] = (round(body_parts["right_wrist"]["x"]), round(body_parts["right_wrist"]["y"]))
            person_data["right_elbow"] = (round(body_parts["right_elbow"]["x"]), round(body_parts["right_elbow"]["y"]))
            person_data["right_shoulder"] = (
                round(body_parts["right_shoulder"]["x"]), round(body_parts["right_shoulder"]["y"]))
            person_data["left_wrist"] = (round(body_parts["left_wrist"]["x"]), round(body_parts["left_wrist"]["y"]))
            person_data["left_elbow"] = (round(body_parts["left_elbow"]["x"]), round(body_parts["left_elbow"]["y"]))
            person_data["left_shoulder"] = (
                round(body_parts["left_shoulder"]["x"]), round(body_parts["left_shoulder"]["y"]))
            person_data["left_hip"] = (round(body_parts["left_hip"]["x"]), round(body_parts["left_hip"]["y"]))
            person_data["left_knee"] = (round(body_parts["left_knee"]["x"]), round(body_parts["left_knee"]["y"]))
            person_data["left_ankle"] = (round(body_parts["left_ankle"]["x"]), round(body_parts["left_ankle"]["y"]))
            person_data["right_hip"] = (round(body_parts["right_hip"]["x"]), round(body_parts["right_hip"]["y"]))
            person_data["right_knee"] = (round(body_parts["right_knee"]["x"]), round(body_parts["right_knee"]["y"]))
            person_data["right_ankle"] = (round(body_parts["right_ankle"]["x"]), round(body_parts["right_ankle"]["y"]))
            person_data["location"] = person_info["location"]
            person_data["center"] = (round(person_data["location"]["left"] + person_data["location"]["width"] / 2),
                                     round(person_data["location"]["top"] + person_data["location"]["height"] / 2))
            gesture_index = judge_gesture(person_data)
            pose_index = judge_pose(person_data)
            person_data['gesture'] = gestures[gesture_index] if gesture_index != -1 else 'idle'
            person_data['gesture_index'] = gesture_index if gesture_index != -1 else -1
            person_data['pose'] = gestures[pose_index] if pose_index != -1 else 'none'
            person_data['pose_index'] = pose_index if pose_index != -1 else -1
            result_data.append(person_data)
        analysis_result = result_data
    else:
        analysis_result = []


def integrate_results(attr_results, analysis_results):
    final_result = []
    for a_result in attr_results:
        for as_result in analysis_results:
            if abs(a_result["center"][0] - as_result["center"][0]) < 50:
                t_result = as_result.copy()
                t_result["gender"] = a_result["gender"]
                t_result["age"] = a_result["age"]
                t_result["upper_color"] = a_result["upper_color"]
                t_result["glasses"] = a_result["glasses"]
                t_result["face_mask"] = a_result["face_mask"]
                t_result["headwear"] = a_result["headwear"]
                final_result.append(t_result)
                break
    return final_result


def distance_filter(results, depth_image):
    filtered_results = []
    for result in results:
        coordinate = depth_image[int(result['location']['top'] + result['location']['height'] / 2)][int(result['location']['left'] + result['location']['width'] / 2)]
        distance = sum([v ** 2 for v in coordinate]) ** 0.5
        if distance < 5000:
            filtered_results.append(result)
    return filtered_results


def judge_gesture(person_data):
    # 判断是否是向左指
    if abs(person_data['left_wrist'][1] - person_data['left_elbow'][1]) < POINT_Y_THRESHOLD and \
            abs(person_data['left_wrist'][1] - person_data['left_shoulder'][1]) < POINT_Y_THRESHOLD and \
            abs(person_data['left_wrist'][0] - person_data['left_shoulder'][0]) > POINT_X_THRESHOLD:
        return 1
    # 判断是否是向右指
    if abs(person_data['right_wrist'][1] - person_data['right_elbow'][1]) < POINT_Y_THRESHOLD and \
            abs(person_data['right_wrist'][1] - person_data['right_shoulder'][1]) < POINT_Y_THRESHOLD and \
            abs(person_data['right_wrist'][0] - person_data['right_shoulder'][0]) > POINT_X_THRESHOLD:
        return 2
    # 判断是否举左手
    if person_data['left_elbow'][1] - person_data['left_wrist'][1] > HANDS_UP_THRESHOLD:
        return 3
    # 判断是否举右手
    if person_data['right_elbow'][1] - person_data['right_wrist'][1] > HANDS_UP_THRESHOLD:
        return 4
    return -1


def judge_pose(person_data):
    # 判断是否站立
    if abs(person_data['left_hip'][1] - person_data['left_knee'][1]) > STANDING_THRESHOLD or \
            abs(person_data['right_hip'][1] - person_data['right_knee'][1]) > STANDING_THRESHOLD or \
            person_data["location"]["height"] > 3 * person_data["location"]["width"]:
        return 5
    # 判断是否坐着
    elif abs(person_data['left_hip'][1] - person_data['left_knee'][1]) < SITTING_THRESHOLD or \
            abs(person_data['right_hip'][1] - person_data['right_knee'][1]) < SITTING_THRESHOLD:
        return 6
    # 判断是否躺着
    elif abs(person_data['left_hip'][0] - person_data['left_knee'][0]) > LYING_THRESHOLD or \
            abs(person_data['left_knee'][0] - person_data['left_ankle'][0]) > LYING_THRESHOLD or \
            abs(person_data['right_hip'][0] - person_data['right_knee'][0]) > LYING_THRESHOLD or \
            abs(person_data['right_knee'][0] - person_data['right_ankle'][0]) > LYING_THRESHOLD:
        return 7
    return -1


# 根据request查找符合条件的人
def count_person_by_request(results, target_index):
    count = 0
    if target_index == -1:
        return len(results)
    if target_index == 0:
        for result in results:
            if result['gesture_index'] in [3, 4]:
                count += 1
        return count
    if 1 <= target_index <= 4:
        for result in results:
            if result['gesture_index'] == target_index:
                count += 1
        return count
    if 5 <= target_index <= 7:
        for result in results:
            if result['pose_index'] == target_index:
                count += 1
        return count
    if 8 <= target_index <= 13:
        if target_index == 8:
            # male
            for result in results:
                if result['gender'] == '男性':
                    count += 1
            return count
        if target_index == 9:
            # female
            for result in results:
                if result['gender'] == '女性':
                    count += 1
            return count
        if target_index == 10:
            # young male
            for result in results:
                if result['gender'] == '男性' and result['age'] in ['幼儿', '青少年']:
                    count += 1
            return count
        if target_index == 11:
            # young female
            for result in results:
                if result['gender'] == '女性' and result['age'] in ['幼儿', '青少年']:
                    count += 1
            return count
        if target_index == 12:
            # old male
            for result in results:
                if result['gender'] == '男性' and result['age'] in ['青年', '中年', '老年']:
                    count += 1
            return count
        if target_index == 13:
            # old female
            for result in results:
                if result['gender'] == '女性' and result['age'] in ['青年', '中年', '老年']:
                    count += 1
            return count
    return None


def call_back(req):
    target_index = req.index
    count = 0
    while True:
        if count >= 10:
            print(colored("No qualified person!", 'bright red'))
            res = xm_count_peopleResponse()
            res.num = 0
            return res
        count += 1
        capture = camera.get_capture()
        #将图片编码到缓存，并保存到本地
        img_bytes = cv2.imencode('.jpg', capture.color)[1].tobytes()
        #添加线程
        attr_thread = threading.Thread(target=deal_with_bodyAttr_result, args=(img_bytes,))
        analysis_thread = threading.Thread(target=deal_with_bodyAnalysis_result, args=(img_bytes,))
        attr_thread.start()
        analysis_thread.start()
        attr_thread.join()
        analysis_thread.join()
        final_results = distance_filter(integrate_results(attr_result, analysis_result), capture.transformed_depth_point_cloud)
        print("Detected People Number is {}.".format(len(final_results)))
        if len(final_results) == 0:
            continue
        drawn_image = drawDebugImage(capture.color, final_results)
        if is_debug:
            cv2.imshow("Detected Image", drawn_image)
            if cv2.waitKey(1) in [ord("q"), ord("Q")]:
                cv2.destroyAllWindows()
                exit(0)
        # 打印debug信息
        for result in final_results:
            print(colored(result, 'purple'))
        result = count_person_by_request(final_results, target_index)
        if result:
            print(colored('count is {}'.format(result), 'purple'))
            res = xm_count_peopleResponse()
            res.num = result
            return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    is_debug = args.debug

    APP_ID = '26806278'
    API_KEY = 'hQbqsjqKoV7Ab5kGlwohqA3c'
    SECRET_KEY = 'KGeEblxK3XT050s3VHf9WgfWPbfEPIHO'
    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
    config = Config(color_resolution=ColorResolution.RES_1080P)
    camera = PyK4A(config)
    camera.start()
    rospy.init_node('countPeopleNode')
    service = rospy.Service('countPeople', xm_count_people, call_back)
    rospy.loginfo('GPSR Count Person\'s Vision Server Start!')
    rospy.spin()
