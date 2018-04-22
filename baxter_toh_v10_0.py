#Allows Baxter to solve the Tower of Hanoi puzzle

#cd ros_ws
#. baxter.sh (remember to check IP address is correct)

#rosrun baxter_tools enable_robot.py -e
#rosrun baxter_tools tuck_arms.py -u

#rosnode list
#rosnode ping <node>
#rosnode kill <node>

#from baxter_toh_v10_0 import BaxterToH

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

#GLOBAL VARIABLES
BLUE = '\033[94m'   #prints to command line in BLUE
WHITE = '\033[0m'   #prints to command line in WHITE

class BaxterToH(object):
    """
    Allows Baxter to solve the Tower of Hanoi puzzle with 3 pegs and any number
    of disks.
    """
    def __init__(self):
        #class attributes
        self.init_x = 0.48   #initial arm x
        self.init_y = 0.67   #initial arm y
        self.init_z = 0.15   #initial arm z
        self.x = self.init_x    #current arm x
        self.y = self.init_y    #current arm y
        self.z = self.init_z    #current arm z
        self.gripper = None #Baxter's left gripper object
        self.grip_state = 100 #gripper position (0 closed, 100 open)
        self.img = None     #current camera image
        self.disk_cols = [] #stores disk colours from big to small (LAB colour space)
        self.num_disks = 3 #number of disks
        self.peg_col = []   #stores peg colour (LAB colour space)
        self.moves = []     #arm moves - [<obj_type>,<obj_id>,<height>] in each row
        self.peg_num_disks = [self.num_disks,0,0]    #number of disks on each peg
        self.num_pegs = 3   #number of pegs for Tower of Hanoi puzzle
        self.table_z = -0.23   #table z coordinate in m
        self.peg_height = 0.05   #peg height in m
        self.offset = [42,-57] #offset between camera and gripper
        self.img_thres = 20
        self.move_inc = 0.005
        self.peg_col_close = []
        #self.peg_col_close = [164.0, 175.0, 137.5]
        #self.picked_up_disk = False
        #self.above_peg = False
        #self.initial_jump_completed = False #Checks if the large jump for find spread disk has been completed
        #self.spread_disk_num = 0 #Keeps track of the number of spread disks 
        #self.state = 'Initial_State'
        self.scale = 0.359/321    #scale {m/pixel}
        self.disk_height = 0.009    #disk thickness
        self.move_to_height = self.table_z+self.peg_height+0.1
        self.drop_height = self.table_z+self.peg_height+0.02
        self.close_col_thresh = 400
        
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
            img_bgr = bridge.imgmsg_to_cv2(msg, "bgr8") #ROS Image msg to OpenCV2
            self.img = img_bgr
            
        #define ROS node and camera topic to subscribe to
        node_name = 't5_tower_of_hanoi'
        topic_name = "/cameras/left_hand_camera/image"
        #topic_name = "t5_cam_throttle"
        
        #setup node
        rospy.init_node(node_name)
        print("Started node: "+node_name)
        
        #subscribe to camera feed
        rospy.Subscriber(topic_name, Image, cam_callback)
        print("Subscribed to topic: "+topic_name)
        
        #set arm to initial position
        print("Initialising arm position")
        self.move_to(self.init_x,self.init_y,self.init_z)
        
        #setup gripper
        print("Initialising gripper")
        self.gripper = baxter_interface.Gripper('left') #Baxter's left gripper object
        self.gripper.calibrate()    #must be calibrated before use
        self.gripper.set_holding_force(30)   #0<holding_force<100
        self.gripper.command_position(self.grip_state)
        
        #wait to acquire first images
        self.update_img()

        #if disk_cols or peg_col is empty, initialise disk/peg colours and number of disks
        if ( (not self.disk_cols) or (not self.peg_col) ):
            print("Colours not selected. Selecting...")
            self.disk_cols = self.choose_colours(self.img)
            self.num_disks = len(self.disk_cols)
            self.peg_col = self.choose_colours(self.img)
            
        #if movements not initialised, compute movements
        if not self.moves:
            print("Moves not initialised. Initialising...")
            #generate setup disk movements (move disk i to peg 0)
            #for i in range(0,self.num_disks):
            #    self.moves.append(['disk',i,0])
            #    self.moves.append(['peg',0,self.peg_num_disks[0]])
            #    self.peg_num_disks[0] = self.peg_num_disks[0] + 1
            #compute disk movements from Tower of Hanoi algorithm
            self.moves, tmp = self.moveTower(self.num_disks, 0, 2, 1, 
                self.moves, self.peg_num_disks)
            #print moves
            print("Moves:")
            for move in self.moves:
                print(move)
                
        #initialise peg positions
        print('Initialising peg positions...')
        self.peg_pos = self.init_peg_pos()
        
        #finish initialisation
        print("Finished initialisation.")
        
    def setup_disks(self):
        """
        Setup Tower of Hanoi puzzle by stacking disks on the fist peg (peg[0])
        """
        disk_cols = self.disk_cols
        img = self.img
        img_shape = img.shape[0:2]
        img_shape = img_shape[::-1] #reverse so img_shape[0]= x dim, [1] = y dim
        cur_arm_pos = [self.x, self.y]
        img_centre = tuple(ij/2 for ij in img_shape)
        offset = self.offset
        des_img_pos = list(map(operator.add, img_centre, offset))
        img_thres = self.img_thres
        peg_pos = self.peg_pos
        move_z = self.move_to_height
        init_x = self.init_x
        init_y = self.init_y
        init_z = self.init_z
        
        print('Setting up disks:')
        for disk_col in disk_cols:
            #find disk in image
            disk_img_pos, img_bin = self.find_colours(img, disk_col)
            print('    Disk found at: '+str(disk_img_pos))
            
            #move arm to disk
            print('    Moving arm...')
            cur_arm_pos = self.move_to_object(disk_img_pos, img_shape, 
                disk_col, des_img_pos, img_thres)
            
            #pickup disk
            print('    Picking up disk...')
            self.pick(0)
            
            #move to peg 0
            print('    Moving to peg 0...')
            self.move_to(peg_pos[0][0], peg_pos[0][1], move_z)
            
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
        moves = self.moves
        peg_pos = self.peg_pos
        move_z = self.move_to_height
        
        print('Solving Tower of Hanoi:')
        for i, move in enumerate(moves):
            des_peg = move[0]
            des_peg_pos = peg_pos[des_peg]
            
            #move to peg
            print('    Moving to peg: '+str(des_peg)+' at: '+str(des_peg_pos))
            self.move_to(des_peg_pos[0], des_peg_pos[1], move_z)
            
            #if index is even, pickup disk, else drop disk
            if i % 2 == 0:
                print('    Picking up disk at height: '+str(move[1]))
                self.pick(move[1])
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
        
    def move_to(self, x_pos, y_pos, z_pos):
        """
        Moves Baxter's left arm to the given position. Maintains a downward 
        pointing gripper pose.
        """
        def ik_angles(X_Pos,Y_Pos,Z_Pos,Roll,Pitch,Yaw):
            """
            Compute the joint angles needed to place the robot arm in a given pose.
            """
            limb_side = 'left'
            ns = "ExternalTools/" + limb_side + "/PositionKinematicsNode/IKService"
            iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
            ikreq = SolvePositionIKRequest()
            hdr = Header(stamp=rospy.Time.now(), frame_id='base')
            quat = tf.transformations.quaternion_from_euler(float(Roll),float(Pitch),float(Yaw))
            poses = {
                'left': PoseStamped(
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

        #move left limb to position
        limb = baxter_interface.Limb('left')
        limb.move_to_joint_positions(angles)
        
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
                tmp, img_bin = self.find_colours(img, [l,a,b], 3)
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
                if len(selected_cols) == 1:
                    return selected_cols[0]
                else:
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
        if len(centroids) == 1:
            return centroids[0], img_bin
        else:
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
            
        #define initial values
        peg_pos = []
        init_img = self.img
        peg_col = self.peg_col
        num_pegs = self.num_pegs
        cur_arm_pos = [self.x, self.y]
        offset = self.offset
        img_thres = self.img_thres
        #x and y dimensions of image
        img_shape = init_img.shape[0:2]
        img_shape = img_shape[::-1] #reverse so img_shape[0]= x dim, [1] = y dim
        img_centre = tuple(ij/2 for ij in img_shape)
        des_img_pos = list(map(operator.add, img_centre, offset))
        move_inc = self.move_inc
        window_name = 'Peg position initialisation'
        
        #get image coordinates of all pegs in initial image
        peg_img_pos, img_bin = self.find_colours(init_img, peg_col, num_pegs)
        print('    Pegs found at img pos: '+str(peg_img_pos))
        
        #loop over pegs in image
        for ith_peg_img_pos in peg_img_pos:
            cur_arm_pos = self.move_to_object(ith_peg_img_pos, img_shape, 
                peg_col, des_img_pos, img_thres)
            
            #record final position
            peg_pos.append(cur_arm_pos)
            print('    Actual peg position: '+str(cur_arm_pos))
        
        print('Finished peg initialisation.')
        cv2.destroyWindow(window_name)  #close image window
        self.move_to(self.init_x, self.init_y, self.init_z) #return to init pos
        return peg_pos
        
    def move_to_object(self, obj_img_pos, img_shape, obj_col, des_img_pos, img_thres):
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
        init_arm_pos = [self.init_x, self.init_y]
        scale = self.scale
    
        print('    Current obj img pos: '+str(obj_img_pos))
        
        #compute desired arm position
        des_arm_pos = self.world_pos_from_img_pos(obj_img_pos, 
                                            img_shape, init_arm_pos, scale)
        print('    Desired arm position: '+str(des_arm_pos))
        
        #move arm to approximate position
        cur_arm_pos = self.move_to(des_arm_pos[0], des_arm_pos[1], 
            self.move_to_height)
        new_img = self.update_img() #wait to update image
        
        #select new colour
        peg_col_close = self.choose_colours(new_img)
        
        #refine position
        new_img_pos, img_bin = self.find_colours(new_img, peg_col_close, 
            num_objects=1, ab_dist_thresh=col_thresh)
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
                
            #show binary image
            show_binary(img_bin, des_img_pos, new_img_pos, img_thres)
            
        return cur_arm_pos
        
    def world_pos_from_img_pos(self, img_pos, img_shape, arm_pos, scale):
        """
        Returns the world coordinates corresponding to given image coordinates
        and the camera position
        """
        centre_x = img_shape[0]/2
        centre_y = img_shape[1]/2
        #scale = 0.2*2/centre_x        #m/pixel
        #print("centre x, y")
        #print(centre_x)
        #print(centre_y)
        
        wld_x = arm_pos[0]
        wld_y = arm_pos[1]
        
        img_x = img_pos[0]
        img_y = img_pos[1]
        #print("img x, y")
        #print(img_x)
        #print(img_y)
        
        img_dx = img_x - centre_x
        img_dy = img_y - centre_y
        #print("img dx, dy")
        #print(img_dx)
        #print(img_dy)
        
        # +wld_x = -img_y ; +wld_y = -img_x
        wld_dx = -img_dy*scale
        wld_dy = -img_dx*scale

        #limit output
        #wld_dx = max(wld_dx, -centre_y*scale)
        #wld_dx = min(wld_dx, centre_y*scale)
        #wld_dy = max(wld_dy, -centre_x*scale)
        #wld_dy = min(wld_dy, centre_x*scale)
        
        new_wld_x = wld_x + wld_dx
        new_wld_y = wld_y + wld_dy
        
        return [new_wld_x, new_wld_y]

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
        self.move_to(new_wld_x, new_wld_y, self.move_to_height)
        
        #return new arm position
        return [new_wld_x, new_wld_y]
        
    def pick(self, obj_height_from_table):
        """
        Pick up an object directly below the arm located at a given height off 
        the table.
        """
        init_x = self.x
        init_y = self.y
        init_z = self.z
        obj_z = self.table_z + obj_height_from_table*self.disk_height
        
        #open gripper
        self.gripper.command_position(100)
        
        #drop to given height
        self.move_to(init_x, init_y, obj_z)
        
        #close gripper
        self.gripper.command_position(0)
        
        #return to initial position
        self.move_to(init_x, init_y, init_z)
    
    def drop(self):
        """
        Drop up an object directly below the arm from a fixed height.
        """
        init_x = self.x
        init_y = self.y
        init_z = self.z
        drop_z = self.drop_height
        
        #drop to given height
        self.move_to(init_x, init_y, drop_z)
        
        #open gripper
        self.gripper.command_position(100)
        
        #return to initial position
        self.move_to(init_x, init_y, init_z)
        
        
        
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
    #num_disks = 3
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
