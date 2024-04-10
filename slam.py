# Pratik Chaudhari (pratikac@seas.upenn.edu)

import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))
       
        # binarized map and log-odds
       
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)
        s.graph = np.zeros(s.cells.shape, dtype=np.int64)
        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        # print (x, y)  # Print the input x and y coordinates
        # print (s.xmin, s.xmax, s.ymin, s.ymax)  # Print the map boundaries
        # print (s.szx, s.szy)  # Print the size of the grid in x and y directions

        # Compute the relative position of the coordinates within the map boundaries
        x_value = x - s.xmin
        y_value = y - s.ymin

        # Convert the relative positions to grid cell indices
        X = x_value / s.resolution
        Y = y_value / s.resolution
        # print (X, Y)  # Print the calculated X and Y values

        # Clip the indices to ensure they are within the valid range of grid cells
        X_final = np.clip(np.abs(X), 0, s.szx - 1)
        Y_final = np.clip(np.abs(Y), 0, s.szy - 1)
        # print (X_final, Y_final)  # Print the clipped X and Y values

        # Create an array of grid cell indices
        array = np.array([X_final, Y_final])
        return array.astype(int)  # Return the array of grid cell indices

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-8*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = 1e-5*np.eye(3)
        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        src_dir = "C:/D/2024/Spring Sem courses/ESE 650 Learning in robo/hw3/hw_export/hw3/p2/p2/p2/"
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX
        num_particles = p.shape[1]

        # generate a random variable in [0, 1/n]
        r = np.random.rand()

        # generate random samples / distances along roulette wheel
        u = (r + np.arange(0, num_particles)) / num_particles

        c = 0
        i = 0
        new_idx = np.zeros_like(u)

        # while(i < num_particles):
        for j, uj in enumerate(u):
            if (c == 0):
                c += w[i]
            elif (uj > c):
                c += w[i]
                i += 1
            new_idx[j] = i
        
        # print(new_idx)
        # new particles
        new_p = p[:, new_idx.astype(int)]
        new_w = np.ones(num_particles)/float(num_particles)
        # print(new_p)
        return new_p, new_w

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX
        

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data

        # 1. from lidar distances to points in the LiDAR frame

        # 2. from LiDAR frame to the body frame

        # 3. from body frame to world frame
        
        inx = np.where((d >= s.lidar_dmin) & (d <= s.lidar_dmax))
        a = s.lidar_angles[inx]
        d = d[inx]
        dist_x = d * np.cos(a)
        dist_y = d * np.sin(a)
        dist_X = dist_x.reshape(1, -1)
        dist_Y = dist_y.reshape(1, -1)
     
        lidar = np.vstack( (np.vstack((dist_X,dist_Y)),np.zeros_like(dist_X)) )
        #changing the frames now
        #lidar to body frame and then to world frame
        homo1 = euler_to_se3(0, head_angle, neck_angle, np.array([0, 0, s.head_height]))
        homo2 = euler_to_se3(0,0,p[-1],np.array([p[0],p[1],s.head_height]))
        
        lidar = np.vstack((lidar, np.ones_like(dist_X)))
        world = np.dot(homo2, np.dot(homo1, lidar))
        
        #need to avoid the ground detection which could be considered as obstacles
        world = world[:, world[1] >= 0.05]
        
        world_final = world[:2, :]
        
        return world_final

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        p_at_t = s.lidar[t]['xyth']
        p_at_t_minus_1 = s.lidar[t-1]['xyth']
        control = smart_minus_2d(p_at_t, p_at_t_minus_1)
        return control

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        no_particles = s.p.shape[1]
        u = s.get_control(t)
        q = np.zeros((s.Q.shape[0],))
        epsilon = np.random.multivariate_normal(q,s.Q,no_particles)
        
        updated_particles = []
        for i in s.p.T:
            val = smart_plus_2d(i, u)
            updated_particles.append(val)
        s.p= np.array(updated_particles).T
        
        for j in range(no_particles):
            updated_particles_w_epsilon = smart_plus_2d(s.p[:, j], epsilon[j, :])   
            s.p[:, j] = updated_particles_w_epsilon
            
    def line_between_points(s, p0, p1):
        # This function implements Bresenham's line algorithm to draw a line between two points on a grid
        x0, y0 = p0.flatten()
        x1, y1 = p1.flatten()
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) >= abs(dy):
            # Drawing a line using y = mx + c
            if dx == 0:
                # Case when p0 == p1
                X = np.array([x0])
                Y = np.array([y0])
            else:
                slope = dy / dx
                c = y0 - slope * x0
                if dx > 0:
                    # Line to the right
                    X = np.arange(x0, x1 + 1)
                elif dx < 0:
                    # Line to the left
                    X = np.arange(x0, x1 - 1, -1)
                Y = np.round(slope * X + c)
        else:
            # Drawing a line using x = my + c
            slope = dx / dy
            c = x0 - slope * y0
            if dy > 0:
                # Line upwards
                Y = np.arange(y0, y1 + 1)
            elif dy < 0:
                # Line downwards
                Y = np.arange(y0, y1 - 1, -1)
            X = np.round(slope * Y + c)
        X_final = X.astype(int)
        Y_final = Y.astype(int)
        return X_final, Y_final
        
            
    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        w_initial= np.log(w) + obs_logp
        w_initial -= slam_t.log_sum_exp(w_initial)
        w_final = np.exp(w_initial)
        return w_final

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX

        d = s.lidar[t]['scan']
        
        tstpm = s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])
        neck_angle, head_angle = s.joint['head_angles'][:, tstpm].flatten()
        
                #First Project Lidar Scan into world Frame
        log_probs = np.zeros(s.p.shape[1])
        for i,p in enumerate(s.p.T):
            occ_cells = s.rays2world(p.T,d,head_angle=head_angle,neck_angle=neck_angle)
            # calculate the observation log-probability
            idx = s.map.grid_cell_from_xy(occ_cells[0,:],occ_cells[1,:])
            log_probs[i] = np.sum(s.map.cells[idx[0],idx[1]])
        #Update the particle weights using observation log-probability
        s.w = s.update_weights(s.w,log_probs)

        #Find the particle with the largest weight 
        idx = int(np.argmax(s.w))
        largest_wgt_pt = s.p[:,idx]
        s.instantaneous = largest_wgt_pt

        #use its occupied cells to update the map.log_odds and map.cells.
        occupied = s.rays2world(largest_wgt_pt,d,head_angle=head_angle,neck_angle=neck_angle)
        cells = s.map.grid_cell_from_xy(occupied[0,:],occupied[1,:]).T
        s.map.log_odds[cells[:,0],cells[:,1]] += s.lidar_log_odds_occ*15
        p_cell = s.map.grid_cell_from_xy(largest_wgt_pt[0],largest_wgt_pt[1])
        s.map.num_obs_per_cell[cells[:,0],cells[:,1]] = 1

        
        freespace = np.array([0])
        for i in range(cells.shape[0]):
            free_x,free_y = s.line_between_points(p_cell,cells[i,:])
            if freespace.shape[0]==1:
                freespace = np.array([free_x[1:-2],free_y[1:-2]])
            else:
                freespace = np.column_stack((freespace,np.array([free_x[1:-2],free_y[1:-2]])))
        #Unique Pixels to take since we dont want to bias by the log odds
        arr = np.unique(freespace.T, axis=0)

        # Check if arr is not empty and contains more than one element before updating the map
        if len(arr) > 1:
            try:
                # Reshape arr to have two columns
                arr = arr.reshape(-1, 2)

                # Update the map only if there are valid cells
                s.map.log_odds[arr[:, 0], arr[:, 1]] += s.lidar_log_odds_free * 5

                # Clip
                s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)
                s.map.cells[s.map.log_odds >= s.map.log_odds_thresh] = 1
                s.map.cells[s.map.log_odds < s.map.log_odds_thresh] = 0
                # The sole job of this is to map the brown area and increase computation time
                s.map.num_obs_per_cell[arr[:, 0], arr[:, 1]] = 1
            except ValueError:
                # Skip map update if arr is empty or contains only one element
                print("Skipping map update due to no valid cells found.")
        else:
            # Skip map update if arr is empty or contains only one element
            print("Skipping map update due to no valid cells found.")

        # Continue with further calculations
        s.resample_particles()
 
               
    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
            
