#Allows Baxter to solve the Tower of Hanoi puzzle

#cd ros_ws
#. baxter.sh (remember to check IP address is correct)

#rosrun baxter_tools enable_robot.py -e
#rosrun baxter_tools tuck_arms.py -u

#rosnode list
#rosnode ping <node>
#rosnode kill <node>

#from baxter_toh_v10_0 import BaxterToH

#---TO IMPROVE---#
# x make Baxter auto-cailibrate height
# - deal with lighting changes
# x improve px->wld scaling at different heights
# x error checking if colour not found in image
# x error checking if disk not picked up

import numpy as np
import cv2              #assumes version 3.3.0 (see https://docs.opencv.org/3.3.0/)
import time
import rospy            # rospy for the subscriber
from sensor_msgs.msg import Image   # ROS message types
from cv_bridge import CvBridge   # ROS Image message -> OpenCV2 image converter
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
import operator

#ASCII ESCAPE CODES (for coloured terminal output)
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
CYAN = '\033[96m'
WHITE = '\033[97m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
GREY = '\033[90m'
BLACK = '\033[90m'
DEFAULT = '\033[99m'

class BaxterToH(object):
    """
    Allows Baxter to solve the Tower of Hanoi puzzle with 3 pegs and any number
    of disks.
    """
    def __init__(self):
        #class attributes
        self.limb_side = 'left'
        self.init_x = 0.48   #initial arm x
        if self.limb_side == 'left':
            self.init_y = 0.67   #initial arm y
        elif self.limb_side == 'right':
            self.init_y = -0.67   #initial arm y
        self.init_z = 0.15   #initial arm z
        self.x = self.init_x    #current arm x
        self.y = self.init_y    #current arm y
        self.z = self.init_z    #current arm z
        self.img = None     #current camera image
        self.disk_cols = [] #stores disk colours from big to small (LAB colour space)
        self.num_disks = 3 #number of disks
        self.peg_col = []   #stores peg colour (LAB colour space)
        self.moves = []     #arm moves - [<obj_type>,<obj_id>,<height>] in each row
        self.peg_num_disks = [self.num_disks,0,0]    #number of disks on each peg
        self.num_pegs = 3   #number of pegs for Tower of Hanoi puzzle
        self.table_z = -0.24   #table z coordinate in m
        self.peg_height = 0.07   #peg height in m
        self.img_thres = 10
        self.move_inc = 0.005
        self.peg_col_close = []
        self.disk_height = 0.009    #disk thickness
        self.close_col_thresh = 400
        #self.cam_gripper_offset = [-0.016,0.012]    #offset between camera and gripper {m}
        self.cam_gripper_offset = [0.03,-0.02]    #camera -> gripper offset {m}
        self.cam_res = (640,400)
        
        #setup node
        node_name = 't5_tower_of_hanoi'
        rospy.init_node(node_name)
        print("Started node: "+node_name)
        
    def initialise(self):
        """
        Initialises Baxter for the Tower of Hanoi puzzle by doing the following:
            - Creating a ROS node
            - Subscribing to the camera
            - Moving arm to the initial position
            - Initialising the gripper
            - Selecting disk and peg colours
            - Initialising required movements
            - Initialising the peg positions
        """
        def cam_callback(msg):
            """
            Save images captured from camera.
            """
            #cam_window_name = "Baxter Video Feed"
            bridge = CvBridge() #instantiate CvBridge
            self.img = bridge.imgmsg_to_cv2(msg, "bgr8") #ROS Image msg to OpenCV2
            
        print(GREEN+'INITIALISING'+WHITE)
            
        #define ROS node and camera topic to subscribe to
        topic_name = "/cameras/"+self.limb_side+"_hand_camera/image"
        
        #subscribe to camera feed
        rospy.Subscriber(topic_name, Image, cam_callback)
        print("Subscribed to topic: "+topic_name)
        
        #initialise camera
        cam_name = self.limb_side+"_hand_camera"
        cam_controller = baxter_interface.CameraController(cam_name)
        cam_controller.resolution = self.cam_res
        
        #set arm to initial position
        print("Initialising arm position")
        self.move_to(self.init_x,self.init_y,self.init_z)
        
        #setup gripper
        print("Initialising gripper")
        self.gripper = baxter_interface.Gripper(self.limb_side) #Baxter's gripper object
        self.gripper.calibrate()    #must be calibrated before use
        self.gripper.set_holding_force(30)   #0<holding_force<100
        self.gripper.command_position(100)  #initially open
        
        #wait to acquire first images
        self.update_img()

        #if disk_cols or peg_col is empty, initialise disk/peg colours and number of disks
        if ( (not self.disk_cols) or (not self.peg_col) ):
            print("Colours not selected. Selecting...")
            self.disk_cols = self.choose_colours(self.img)
            self.num_disks = len(self.disk_cols)
            self.peg_col = self.choose_colours(self.img)
            self.peg_col = self.peg_col[0]
            self.peg_num_disks = [self.num_disks,0,0]
            
        #if movements not initialised, compute movements
        if not self.moves:
            print("Moves not initialised. Initialising...")
            self.moves, tmp = self.moveTower(self.num_disks, 0, 2, 1, 
                self.moves, self.peg_num_disks)
            #print moves
            print("Moves:")
            for move in self.moves:
                print(move)
                
        #initialise peg positions and table height
        print('Initialising peg positions...')
        self.peg_pos, self.peg_table_z, self.table_z = self.init_peg_pos()
        
        #finish initialisation
        print("Finished initialisation.")
        
    def setup_disks(self):
        """
        Setup Tower of Hanoi puzzle by stacking disks on the fist peg (peg[0])
        """
        disk_cols = self.disk_cols
        img_shape = self.img.shape[0:2]
        img_shape = img_shape[::-1] #reverse so img_shape[0]= x dim, [1] = y dim
        arm_pos = [self.x, self.y, self.z]
        img_centre = tuple(ij/2 for ij in img_shape)
        img_thres = self.img_thres
        peg_pos = self.peg_pos
        init_x = self.init_x
        init_y = self.init_y
        init_z = self.init_z
        x = self.x
        y = self.y
        z = self.z
        move_inc = self.move_inc
        table_z = self.table_z
        disk_height = self.disk_height
        
        #move_to_disk_z = self.table_z + 0.15
        move_to_peg_z = self.table_z + self.peg_height + 0.03
        move_to_disk_z = move_to_peg_z
        disk_z = table_z + disk_height
        
        #move to inital position
        self.move_to(init_x, init_y, init_z)
        
        print(GREEN+'SETUP_DISKS'+WHITE)
        for disk_col in disk_cols:
            #loop until disk successfully picked up
            picked_up = False
            while not picked_up:
                #cature new image
                img = self.update_img()
                
                #find disk in image
                disk_img_pos, img_bin = self.find_colours(img, disk_col)
                disk_img_pos = disk_img_pos[0]
                print('    Disk found at: '+str(disk_img_pos))
                
                #get current arm position
                arm_pos = [self.x, self.y, self.z]
                
                #move arm to disk
                print('    Moving arm...')
                self.move_to_object(disk_img_pos, img_shape, img_thres, arm_pos, 
                    disk_z, move_to_disk_z)
                    
                #drop to table
                print('    Descending...')
                self.move_to_table(table_z, reset_pos=True, close_gripper=True)
                
                #read gripper pos
                position = self.gripper.position()
                if position>5:
                    picked_up = True
                else:
                    #if not picked up, update image, recalibrate table height
                    self.gripper.command_position(100)
                    print('    Missed disk. Reattempting...')
            
            #move to peg 0
            print('    Moving to peg 0...')
            self.move_to(self.x, self.y, move_to_peg_z) #move vertically up first
            self.move_to(peg_pos[0][0], peg_pos[0][1], move_to_peg_z)
            
            #drop disk
            print('    Dropping disk...')
            self.drop()
            
            #return to neutral position
            print('    Moving to initial position...')
            self.move_to(init_x, init_y, init_z)
        print('Finished setting up disks.')
            
    def solve_puzzle(self):
        """
        Perform moves to solve Toware of Hanoi puzzle
        """
        init_x = self.init_x
        init_y = self.init_y
        init_z = self.init_z
        moves = self.moves
        peg_pos = self.peg_pos
        move_to_peg_z = self.table_z + self.peg_height + 0.02
        peg_table_z = self.peg_table_z
        disk_height = self.disk_height
        
        self.move_to(init_x, init_y, init_z)
        
        print(GREEN+'SOLVING PUZZLE'+WHITE)
        for i, move in enumerate(moves):
            des_peg = move[0]
            des_peg_pos = peg_pos[des_peg]
            
            #move to peg
            print('    Moving to peg '+str(des_peg)+' at: '+str(des_peg_pos))
            self.move_to(des_peg_pos[0], des_peg_pos[1], move_to_peg_z)
            
            #compute required height for pick up
            pick_up_z = peg_table_z[des_peg] + move[1]*disk_height
            
            #if index is even, pickup disk, else drop disk
            if i % 2 == 0:
                print('    Picking up disk at height: '+str(move[1]))
                picked_up = False
                while not picked_up:
                    picked_up = self.pick(pick_up_z,'m')
                    if not picked_up:
                        print('    Missed disk. Reattempting...')
                        pick_up_z = pick_up_z-0.005
            else:
                print('    Dropping disk')
                self.drop()
        print('Finished solving puzzle')
        
    def run(self):
        """
        Executes Tower of Hanoi solution using Baxter.
        """
        self.initialise()
        self.setup_disks()
        self.solve_puzzle()
        input('Finished. Press ENTER to exit.')
        
    def move_to(self, x_pos, y_pos, z_pos, blocking=True):
        """
        Moves Baxter's arm to the given position. Maintains a downward 
        pointing gripper pose.
        """
        def ik_angles(X_Pos,Y_Pos,Z_Pos,Roll,Pitch,Yaw):
            """
            Compute the joint angles needed to place the robot arm in a given pose.
            """
            limb_side = self.limb_side
            ns = "ExternalTools/" + limb_side + "/PositionKinematicsNode/IKService"
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
		                )
                    )
                )
            }

            ikreq.pose_stamp.append(poses[limb_side])
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
                print("INVALID POSE - No Valid Joint Solution Found.")

            return 0
            
        roll = 0
        pitch = 3.14
        yaw = 0     #controls roll of gripper

        #compute required joint angles
        angles = ik_angles(x_pos,y_pos,z_pos,roll,pitch,yaw)

        #move limb to position
        self.limb = baxter_interface.Limb(self.limb_side)
        if blocking:
            self.limb.move_to_joint_positions(angles)
        else:
            self.limb.set_joint_positions(angles)
        
        #update current position
        self.x = x_pos
        self.y = y_pos
        self.z = z_pos
        
        return [x_pos, y_pos]
        
    def update_img(self):
        """
        Returns an updated image
        """
        img_old = self.img
        #Ensure at least one image has been captured
        attempts = 0
        while img_old == None:
            print("Wating to capture first image...")
            time.sleep(1)
            img_old = self.img
            attempts = attempts + 1
            if attempts == 10:
                raise Exception('No images captured after 10 attempts, aborting.')
        img_new = img_old
        #wait until new image is captured
        print('Waiting for new image...')
        while np.all(img_old == img_new):
            time.sleep(0.1)
            img_new = self.img
        print('New image acquired.')
        
        return img_new

    def choose_colours(self, img):
        """
        Returns an array containing the selected colours in the LAB colour space.
        """
        def choose_colours_callback(event, x, y, flags, param):
            """
            Mouse callback function for choose_colours()
            """
            window_name_callback = "Select colours:"
            nhood = 10   #neighbourhood to take median
            lim = nhood/2
            
            #if left mouse button clicked, save cursor position
            if event == cv2.EVENT_LBUTTONDOWN:
                #take median colour in nhood*nhood neighbourhood
                l = np.median(img_lab[ y - lim : y + lim+1 , x - lim : x + lim , 0])
                a = np.median(img_lab[ y - lim : y + lim+1 , x - lim : x + lim , 1])
                b = np.median(img_lab[ y - lim : y + lim+1 , x - lim : x + lim , 2])
                selected_cols.append([l, a, b])
                #display 3 largest objects with selected colours in binary image
                tmp, img_bin = self.find_colours(img, [l,a,b], num_objects=3)
                cv2.imshow(window_name_callback,img_bin)
                cv2.waitKey(1000) & 0xFF
                cv2.destroyWindow(window_name_callback)
                
        window_name = "Click on objects to select colours."
        selected_cols = []
        
        #convert BGR to LAB
        img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

        #create window, set callback function
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name,choose_colours_callback)
        
        #while ENTER or BACKSPACE not pressed, wait for mouse clicks
        key = -1
        while key != 13:
            cv2.imshow(window_name,img)
            key = cv2.waitKey(0) & 0xFF
            #if ENTER pressed and colours selected, save colours and exit
            if key==13:
                print("Selected colours:")
                for col in selected_cols:
                    print("    "+str(col))
                cv2.destroyWindow(window_name)
                return selected_cols
            #if BACKSPACE pressed, delete last entry and continue
            elif key==8:
                del(selected_cols[-1])
                print("Removed last entry.")
        
    def find_colours(self, img, colour, num_objects=1, ab_dist_thresh=50):
        """
        Returns the position and binary image of objects in a given image 
        matching a given colour.
        """
        img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)   #convert to LAB colour space    
        img_a = img_lab[:,:,1]                          #A compononent of image
        img_b = img_lab[:,:,2]                          #B compononent of image  
        des_a = colour[1]                               #A component of desired colour
        des_b = colour[2]                               #B component of desired colour
        
        #Compute difference between desired components and actual components
        d_a = img_a - des_a
        d_b = img_b - des_b
        dist_squared = d_a**2 + d_b**2
        
        #Apply threshold
        img_bin = np.uint8(dist_squared<ab_dist_thresh)*255
        
        #do connected components analysis to find centroids of large connected objects
        conn_comp = cv2.connectedComponentsWithStats(img_bin, 8, cv2.CV_32S)

        #sort by area, from largest to smallest
        areas = np.int_(conn_comp[2][:,4])
        idx = areas.argsort()
        idx = idx[::-1]
        centroids = np.int_(conn_comp[3])
        centroids = centroids[idx[1:num_objects+1]]
        
        #if more than one object returned, order from left to right
        idx = centroids[:,0].argsort()   #sort by x value
        centroids = list(centroids[idx])
        
        #return centroid position and binary image of detected objects
        return centroids, img_bin
        
    def moveTower(self, height, fromPole, toPole, withPole, moves, peg_num_disks):
        """
        Returns the moves to solve the Tower of Hanoi puzzle with 3 pegs.
        """
        def moveDisk(fromPole, toPole):
            peg_num_disks[fromPole] = peg_num_disks[fromPole] - 1
            moves.append([ fromPole, peg_num_disks[fromPole] ])
            moves.append([ toPole, peg_num_disks[toPole] ])
            peg_num_disks[toPole] = peg_num_disks[toPole] + 1
            return moves, peg_num_disks
            
        if height >= 1:
            moves, peg_num_disks = self.moveTower(height-1, fromPole, withPole, 
                toPole, moves, peg_num_disks)
            moves, peg_num_disks = moveDisk(fromPole, toPole)
            moves, peg_num_disks = self.moveTower(height-1, withPole, toPole, 
                fromPole, moves, peg_num_disks)
        return moves, peg_num_disks

    def init_peg_pos(self):
        """
        Returns the initial world x, y coordinates of the given number of pegs 
        in a given initial image, taken from a given camera position. Assumes
        camera is looking in the negative z direction.
        """
        window_name = 'Peg position initialisation'
        peg_pos = []
        peg_table_z = []
        init_img = self.img
        peg_col = self.peg_col
        num_pegs = self.num_pegs
        img_thres = self.img_thres
        img_shape = init_img.shape[0:2]
        img_shape = img_shape[::-1] #reverse so img_shape[0]= x dim, [1] = y dim
        init_arm_pos = [self.init_x, self.init_y, self.init_z]
        move_z = self.table_z + self.peg_height + 0.01
        peg_z = self.table_z + self.peg_height
        table_z = self.table_z
        
        #get image coordinates of all pegs in initial image
        peg_img_pos, img_bin = self.find_colours(init_img, peg_col, num_pegs)
        print('    Pegs found at img pos: '+str(peg_img_pos))
        
        #loop over pegs in image
        for ith_peg_img_pos in peg_img_pos:
            #move above peg
            cur_arm_pos = self.move_to_object(ith_peg_img_pos, img_shape, 
                img_thres, init_arm_pos, peg_z, move_z)
                
            #find and record table height at peg
            ith_peg_table_z = self.move_to_table(table_z, reset_pos=True)
            peg_table_z.append(ith_peg_table_z)
            print('    Table z at peg: '+str(ith_peg_table_z))
            
            #record final position
            peg_pos.append(cur_arm_pos)
            print('    Actual peg position: '+str(cur_arm_pos))
            
        #take average of peg_table_z to find table_z
        table_z = np.mean(peg_table_z)
        print('    Peg positions: '+str(peg_pos))
        print('    Peg table z: '+str(peg_table_z))
        print('    Mean table z: '+str(table_z))
        
        print('Finished peg initialisation.')
        cv2.destroyWindow(window_name)  #close image window
        self.move_to(self.init_x, self.init_y, self.init_z) #return to init pos
        
        return peg_pos, peg_table_z, table_z
        
        
    def move_to_object(self, obj_img_pos, img_shape, img_thres, arm_pos, obj_z, 
        move_z):
        """
        Move Baxter's arm to an object and returns the final position.
        """
        def show_binary(img_bin, des_img_pos, new_img_pos, img_thres):
            """
            Show intermediate binary image while refining position.
            """
            img_bgr = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
            #draw tolerance lines
            #left/right vertical lines
            xl = des_img_pos[0] - img_thres
            xr = des_img_pos[0] + img_thres
            y1 = 0
            y2 = img_shape[1]
            cv2.line(img_bgr,(xl,y1),(xl,y2),(0,255,0),1)
            cv2.line(img_bgr,(xr,y1),(xr,y2),(0,255,0),1)
            #top/bottom horizontal lines
            yt = des_img_pos[1] - img_thres
            yb = des_img_pos[1] + img_thres
            x1 = 0
            x2 = img_shape[0]
            cv2.line(img_bgr,(x1,yt),(x2,yt),(0,255,0),1)
            cv2.line(img_bgr,(x1,yb),(x2,yb),(0,255,0),1)
            #draw circle at detected object
            cv2.circle(img_bgr,tuple(new_img_pos),6,(255,0,0),2)
            #show image
            cv2.imshow(window_name, img_bgr)
            cv2.waitKey(1000) & 0xFF
            
        cur_arm_pos = [self.x, self.y]
        move_inc = self.move_inc
        window_name = 'Refine position'
        col_thresh = self.close_col_thresh
        cam_gripper_offset = self.cam_gripper_offset
        
        print('    Current obj img pos: '+str(obj_img_pos))
        
        #compute desired arm position
        des_arm_pos = self.world_pos_from_img_pos(obj_img_pos, img_shape, 
            arm_pos, obj_z)
        print('    Desired arm position: '+str(des_arm_pos))
        
        #move arm to approximate position
        cur_arm_pos = self.move_to(des_arm_pos[0], des_arm_pos[1], move_z)
        new_img = self.update_img() #wait to update image
        
        #select new colour
        peg_col_close = self.choose_colours(new_img)
        peg_col_close = peg_col_close[0]
        
        #compute desired image position
        arm_pos = [self.x, self.y, self.z]
        img_centre_x = img_shape[0]/2
        img_centre_y = img_shape[1]/2
        scale = self.compute_scale(arm_pos[2]-obj_z)
        img_des_x = img_centre_x - int(cam_gripper_offset[1]/scale)
        img_des_y = img_centre_y - int(cam_gripper_offset[0]/scale)
        des_img_pos = [img_des_x,img_des_y]
        print('    Desired img pos: '+str(des_img_pos))
        
        new_img_pos, img_bin = self.find_colours(new_img, peg_col_close, 
            num_objects=1, ab_dist_thresh=col_thresh)
        new_img_pos = new_img_pos[0]
        
        show_binary(img_bin, des_img_pos, new_img_pos, img_thres)
        while ( abs(new_img_pos[0] - des_img_pos[0]) > img_thres or 
                abs(new_img_pos[1] - des_img_pos[1]) > img_thres    ):
            #refine position
            cur_arm_pos = self.move_to_refine(des_img_pos, new_img_pos, 
                cur_arm_pos, move_inc, img_thres)
            
            #update image
            new_img = self.update_img()
            
            #find new image position of peg
            new_img_pos, img_bin = self.find_colours(new_img, peg_col_close, 
                num_objects=1, ab_dist_thresh=col_thresh)
            new_img_pos = new_img_pos[0]
                
            #show binary image
            show_binary(img_bin, des_img_pos, new_img_pos, img_thres)
            
        return cur_arm_pos
        
    def world_pos_from_img_pos(self, img_pos, img_shape, arm_pos, obj_z):
        """
        Returns the world coordinates corresponding to given image coordinates
        and the camera position
        """
        cam_gripper_offset = self.cam_gripper_offset
        
        scale = self.compute_scale(arm_pos[2] - obj_z)
        
        centre_x = img_shape[0]/2
        centre_y = img_shape[1]/2
        
        wld_x = arm_pos[0] - cam_gripper_offset[0]
        wld_y = arm_pos[1] - cam_gripper_offset[1]
        
        img_x = img_pos[0]
        img_y = img_pos[1]
        
        img_dx = img_x - centre_x
        img_dy = img_y - centre_y
        
        # +wld_x = -img_y ; +wld_y = -img_x
        wld_dx = -img_dy*scale
        wld_dy = -img_dx*scale
        
        new_wld_x = wld_x + wld_dx
        new_wld_y = wld_y + wld_dy
        
        return [new_wld_x, new_wld_y]
        
    def compute_scale(self,dist_to_obj):
        """
        Computes pixel->metres scaling factor, given the distance to the object.
        """
        #linear reqression from Excel
        slope =     0.002317
        intercept = 0.000402
        
        scale = slope*dist_to_obj + intercept
        
        return scale
        
    def move_to_refine(self, des_img_pos, act_img_pos, current_world_pos, increment, img_thresh):
        """
        Moves Baxters arm by one increment such that the object at the given 
        actual image position moves to the desired image position
        """
        des_img_x = des_img_pos[0]
        des_img_y = des_img_pos[1]
        act_img_x = act_img_pos[0]
        act_img_y = act_img_pos[1]
        cur_wld_x = current_world_pos[0]
        cur_wld_y = current_world_pos[1]
        new_wld_x = cur_wld_x
        new_wld_y = cur_wld_y
        
        #object to the left -> move left (-wld_y)
        if (act_img_x < des_img_x-img_thresh):
            print('    Moving left')
            new_wld_y = cur_wld_y + increment
        #object to the right -> move right (+wld_y)
        elif (act_img_x > des_img_x+img_thresh):
            new_wld_y = cur_wld_y - increment
            print('    Moving right')
        #object to the top -> move forward (+wld_x)
        if (act_img_y < des_img_y-img_thresh):
            new_wld_x = cur_wld_x + increment
            print('    Moving forward')
        #object to the bottom -> move backward (-wld_x)
        elif (act_img_y > des_img_y+img_thresh):
            new_wld_x = cur_wld_x - increment
            print('    Moving backward')
        
        #move arm to new coordinates
        self.move_to(new_wld_x, new_wld_y, self.z)
        
        #return new arm position
        return [new_wld_x, new_wld_y]
        
    def pick(self, obj_height_from_table, height_units='disks'):
        """
        Pick up an object directly below the arm located at a given height off 
        the table.
        """
        init_x = self.x
        init_y = self.y
        init_z = self.z
        
        #convert input height to metres
        if height_units == 'disks':
            obj_z = self.table_z + obj_height_from_table*self.disk_height
        elif height_units == 'm':
            obj_z = obj_height_from_table
            
        print('    Object at height: '+str(obj_z))
        
        #open gripper
        self.gripper.command_position(100)
        
        #drop to given height
        self.move_to(init_x, init_y, obj_z)
        
        #close gripper
        self.gripper.command_position(0)
        
        #return to initial position
        self.move_to(init_x, init_y, init_z)
        
        #return if gripper successfully picked up disk
        position = self.gripper.position()
        if position>5:
            print("    Picked up object")
            success = True
        else:
            print("    Didn't pick up object")
            self.gripper.command_position(100)
            success = False
        
        return success
    
    def drop(self):
        """
        Drop up an object directly below the arm from a fixed height.
        """
        init_x = self.x
        init_y = self.y
        init_z = self.z
        drop_z = self.table_z+self.peg_height+0.02
        
        #drop to given height
        self.move_to(init_x, init_y, drop_z)
        
        #open gripper
        self.gripper.command_position(100)
        
        #return to initial position
        self.move_to(init_x, init_y, init_z)
        
    def move_to_table(self, table_z_approx, reset_pos=False, close_gripper=False):
        """
        Move arm to table, and return table z value
        """
        force_thresh = 8   #endpoint force above which table hit {N}
        z_inc = 0.005       #increment for decent {m}
        
        #set initial values
        x = self.x
        y = self.y
        z = self.z
        init_x = x
        init_y = y
        init_z = z
        limb_side = self.limb_side
        limb = self.limb
        
        #move close to table
        z = table_z_approx
        self.move_to(x,y,z)
        
        #while endpoint z force>thresh, keep descending
        print('    Descending to table...')
        mag_force_z = abs(0.5 * force_thresh)
        while (mag_force_z < force_thresh):
            z = z - z_inc
            self.move_to(x,y,z, blocking=False)
            time.sleep(1)   #wait for endpoint force to update
            wrench = limb.endpoint_effort()
            mag_force_z = abs(wrench['force'][2])
            print('    z: '+str(z)+', force: '+str(mag_force_z))
        #ascend until no force
        while (mag_force_z > force_thresh):
            z = z + z_inc
            self.move_to(x,y,z, blocking=False)
            time.sleep(1)   #wait for endpoint force to update
            wrench = limb.endpoint_effort()
            mag_force_z = abs(wrench['force'][2])
            print('    z: '+str(z)+', force: '+str(mag_force_z))
            
        table_height = z
        
        if close_gripper:
            self.gripper.command_position(0)
        
        #return to initial position is specified
        if reset_pos:
            self.move_to(init_x,init_y,init_z)
        
        #return table height
        print('    Table z: '+str(table_height))
        return table_height
        
    def save_config(self,name='config.json'):
        """
        Save Baxter initialisation config.
        """
        config_settings = {"HOST":"analog.local"
                            }
        with open(name, 'w') as outfile:
            json.dump(config_settings, outfile)
            
    def load_config(self,name='config.json'):
        """
        Load Baxter initialisation config.
        """
        with open(name) as data_file:    
            config_settings = json.load(data_file)
    
        
        
if __name__ == '__main__':
    bax = BaxterToH()
    bax.run()
    
    #---TEST choose_colours---#
    #img = cv2.imread('1.jpeg')
    #disk_cols = bax.choose_colours(img)
    #peg_col = bax.choose_colours(img)
    #num_disks = len(disk_cols)
    #print(disk_cols)
    #print(peg_col)
    #print(num_disks)
    
    #---TEST moveTower---#
    #moves = []
    #num_disks = 2
    #peg_num_disks=[num_disks,0,0]
    #print("peg_num_disks:")
    #print(peg_num_disks)
    #moves, tmp = bax.moveTower(num_disks, 0, 2, 1, 
    #    moves, peg_num_disks)
    #print("Moves:")
    #for move in moves:
    #    print(move)
    
    #---TEST world_pos_from_img_pos---#
    #img = cv2.imread('1.jpeg')
    #peg_col = bax.choose_colours(img)
    #num_pegs = 3
    #peg_img_pos, img_bin = bax.find_colours(img, peg_col, num_pegs)
    #img_shape = img.shape[0:2]
    #img_shape = img_shape[::-1] #reverse so img_shape[0]= x dim, [1] = y dim
    #for ith_peg_img_pos in peg_img_pos:
    #    arm_pos = [bax.init_x, bax.init_y]
    #    wld_pos = bax.world_pos_from_img_pos(ith_peg_img_pos, img_shape, arm_pos, bax.scale)
    #    print(wld_pos)
    
    #---TEST init_table_height---#
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
