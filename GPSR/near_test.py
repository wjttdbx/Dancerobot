import rospy
from std_srvs.srv import *
from xm_msgs.srv import *
from xm_msgs.msg import *

xm_findobject = rospy.ServiceProxy('get_3position', xm_3ObjectDetect)
xm_findobject.wait_for_service(timeout=30.0)
req = xm_3ObjectDetectRequest()
req.object_name1= 'biscuit'
req.object_name1= 'lays'
req.object_name1= 'orange juice'
res = xm_findobject.call(req)
print(res)
