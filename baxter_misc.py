#cd ros_ws
#. baxter.sh (remember to check IP address is correct)

#rosrun baxter_tools enable_robot.py -e
#rosrun baxter_tools tuck_arms.py -u

#rosnode list
#rosnode ping <node>
#rosnode kill <node>

#for computer vision functions
import numpy as np
import cv2              #assumes version 3.3.0 (see https://docs.opencv.org/3.3.0/)
import time
import rospy            # rospy for the subscriber
from sensor_msgs.msg import Image   # ROS message types
#from std_msg.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError   # ROS Image message -> OpenCV2 image converter

#for inverse kinematics functions
import baxter_interface
import struct
import tf
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
import baxter_external_devices
    
class Baxter(object):
    """
    Display live Baxter left-hand camera feed
    """
    def __init__(self):
        self.limb_side = 'left'
        self.grip_state = 100 #gripper position (0 closed, 100 open)

        #setup node
        node_name = 't5_baxter_misc'
        print("Starting node: "+node_name)
        rospy.init_node(node_name, anonymous=True)

        #setup gripper
        print("Initialising gripper")
        gripper = baxter_interface.Gripper(self.limb_side)
        gripper.calibrate()    #must be calibrated before use
        gripper.set_holding_force(30)   #0<holding_force<100
        gripper.command_position(self.grip_state)
        


    def cam(self):
        def cam_callback(msg):
            """
            Executes on 15Hz timer, from camera subscriber
            """
            window_name = "Baxter Video Feed (Direct)"
            bridge = CvBridge() #instantiate CvBridge
            
            img = bridge.imgmsg_to_cv2(msg, "bgr8") #ROS Image msg to OpenCV2

            #show video feed
            cv2.imshow(window_name,img) 
            key = cv2.waitKey(1000/15) & 0xFF

            #save image if ENTER pressed
            if key == 13:
                cv2.imwrite('camera_image.jpeg', img)
            
        #define ROS node and camera topic to subscribe to
        topic_name = "/cameras/"+self.limb_side+"_hand_camera/image"

        #subscribe to camera feed
        rospy.Subscriber(topic_name, Image, cam_callback)
        print("Subscribed to topic: "+topic_name)

        rospy.spin()
        cv2.destroyAllWindows()



    #not working
    def cam_setup(self):
        cam_name = self.limb_side+"_hand_camera"
        cam_controller = baxter_interface.CameraController(cam_name)
        cam_controller.resolution = (640,400)
        redo = 'y'
        #while (redo == 'y'):
        while (1):
            exposure = cam_controller.exposure
            gain = cam_controller.gain
            wb_g = cam_controller.white_balance_green
            wb_b = cam_controller.white_balance_red
            wb_r = cam_controller.white_balance_blue
            print("Current exposure/gain: "+str(exposure)+"/"+str(gain)
                +"/"+str(wb_r)+"/"+str(wb_g)+"/"+str(wb_b));
            settings = input("    New settings: ")
            cam_controller.exposure = settings[0]
            cam_controller.gain = settings[1]
            cam_controller.white_balance_red = settings[2]
            cam_controller.white_balance_green = settings[3]
            cam_controller.white_balance_blue = settings[4]
            #redo = input("    Redo settings? (y/n) ")



    def move_to(self,x_pos,y_pos,z_pos,yaw=0):
        def ik_angles(X_Pos,Y_Pos,Z_Pos,Roll,Pitch,Yaw):
            limb = self.limb_side
            ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
            iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
            ikreq = SolvePositionIKRequest()
            hdr = Header(stamp=rospy.Time.now(), frame_id='base')
            quat = tf.transformations.quaternion_from_euler(float(Roll),float(Pitch),float(Yaw))
            poses = {
                self.limb_side: PoseStamped(
                    header=hdr,
                    pose=Pose(
                        position=Point(
		                    x=float(X_Pos),
                            y=float(Y_Pos),
                            z=float(Z_Pos),
                        ),
                        orientation=Quaternion(
		                   x = quat[0],
		                   y = quat[1],
		                   z = quat[2],
		                   w = quat[3],
		                ),
                    ),
                ),
            }

            ikreq.pose_stamp.append(poses[limb])
            try:
                rospy.wait_for_service(ns, 5.0)
                resp = iksvc(ikreq)
            except (rospy.ServiceException, rospy.ROSException), e:
                rospy.logerr("Service call failed: %s" % (e,))
                return 1

            # Check if result valid, and type of seed ultimately used to get solution
            # convert rospy's string representation of uint8[]'s to int's
            resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                                       resp.result_type)
            if (resp_seeds[0] != resp.RESULT_INVALID):
                # Format solution into Limb API-compatible dictionary
                limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
                
                return limb_joints
            else:
                #print("INVALID POSE - No Valid Joint Solution Found.")
                return None
                
        
        
        limb = baxter_interface.Limb(self.limb_side)
        
        init_angles = limb.joint_angles()
        
        roll = 0
        pitch = 3.14
        angles = ik_angles(x_pos,y_pos,z_pos,roll,pitch,yaw)

        if angles is None:
            print('INVALID POSE - No Valid Joint Solution Found.')
            return False
        else:
            limb.set_joint_positions(angles)
            time.sleep(3)
            wrench = limb.endpoint_effort()
            force_z = abs(wrench['force'][2])
            print(wrench['force'])
            print(force_z)
            if force_z>10:
                limb.set_joint_positions(init_angles)
                print('Contact force limit exceeded')
            return True
        
        

    def toggle_grip(self):
        gripper = baxter_interface.Gripper(self.limb_side)
    
        if self.grip_state == 0:
            print("Opening gripper")
            self.grip_state = 100
            gripper.command_position(self.grip_state)
        elif self.grip_state == 100:
            print("Closing gripper")
            self.grip_state = 0
            gripper.command_position(self.grip_state)
            time.sleep(0.5)
            position = gripper.position()
            print('    '+str(position))
            if position>5:
                print("    Picked up object")
            else:
                print("    Didn't pick up object")
            
            
            
    def print_endpoint_effort(self):
        limb = baxter_interface.Limb(self.limb_side)
        
        done = False
        while not done and not rospy.is_shutdown():
            c = baxter_external_devices.getch()
            wrench = limb.endpoint_effort()
            print('Force (x,y,z): '+str(wrench['force']))
            time.sleep(0.5)
            if c:
                if c in ['\x1b', '\x03']:   #check if ESC of CTRL-C pressed
                    done = True


    


#for command line execution
if __name__ == '__main__':
    bax = Baxter()
    #bax.move_to(0.48,-0.67,0.2)
    #bax.print_endpoint_effort()
    
    #while 1:
    #    bax.toggle_grip()
    #    time.sleep(2)
    
    #x=0.48
    #y=-0.67
    #z=0.15
    #valid_pose = True
    #while valid_pose:
    #    valid_pose = bax.move_to(x,y,z)
    #    x = x+0.1
    
    bax.cam()
    #bax.cam_setup()











