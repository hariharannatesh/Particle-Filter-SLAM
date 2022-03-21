
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 09:46:04 2022

@author: Hariharan Natesh
"""

import pandas as pd
import cv2
import numpy as np
from pr2_utils import read_data_from_csv
from pr2_utils import bresenham2D
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D

import os
import pr2_utils


def homogenize(Y):
    
    '''
    Converts the input
    to homogenous form.
    
    Y: input vector
    '''
    ones = np.ones(Y.shape[1])
    homo = np.vstack((Y,ones))
    return homo


def plot_map(MAP):
    
    '''
    Plots the occupancy
    map.
    
    MAP : Dictionary
    '''
    
    plt.imshow(MAP['map'],cmap="hot");
    plt.title("Occupancy grid map")
    
    
def plot_texture_map(MAP):
    
    '''
    Plots the texture MAP.
    
    MAP : Dictionary
    '''
    plt.imshow(MAP['texture_map']);
    plt.title("Textue map")
    
def init_map(MAP,lidar_data,angles):
    
    '''
    Funtion to initialize occupancy
    map.
    
    MAP: Dictionary 
    lidar_data: Data from lidar scan
    angles: Angles covered by the lidar.
    
    '''
    ranges = lidar_data[0, :]

    # take valid indices
    indValid = np.logical_and((ranges < 80),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    
    

    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    

    Y = np.stack((xs0,ys0))
    
    
    Y_homo = homogenize(Y)

    R = np.array([[0.00130201,0.796097,0.605167],[0.999999,-0.000419027,-0.00160026],[-0.00102038,0.605169,-0.796097]])
    
    p = np.array([[0.8349,-0.0126869,1.76416]]).T
    
    Y_homo_l2v = np.linalg.inv(R) @ (Y_homo - p)
    
    

    
    Y_l2v = Y_homo_l2v[:2,:].reshape(2,-1)
    
    

    x_coor = Y_l2v[0,:]
    y_coor = Y_l2v[1,:]
    plt.plot(x_coor,y_coor,'.k')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Laser reading")
    plt.axis('equal')
    

    
    xcell = np.ceil((x_coor - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    ycell = np.ceil((y_coor - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    

    begin_x = np.ceil((0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    begin_y = np.ceil((0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    MAP['log_map'][xcell,ycell]+=np.log(9) 

    
    MAP['log_map']  = np.where(MAP['log_map']>50,50,MAP['log_map'])
    
    
    for i in range(Y_l2v.shape[1]):
        ex = xcell[i]
        ey = ycell[i]
        
        bresenham_coords = bresenham2D(begin_x, begin_y, ex, ey)
        x_coords,y_coords = bresenham_coords[0,1:-1],bresenham_coords[1,1:-1]
        MAP['log_map'][x_coords.astype(int),y_coords.astype(int)]-=np.log(9) 

        
    MAP['log_map']  = np.where(MAP['log_map']<-50,-50,MAP['log_map'])
    MAP['map'] = np.exp(MAP['log_map'])/(1+np.exp(MAP['log_map']))
    
    MAP['map'][MAP['map']>0.5] = 1
    MAP['map'][MAP['map']<=0.5] = 0
    plot_map(MAP)



def texture_map(timestamp,x0,y0,yaw):
    '''
    Build a texture map 
    by using the stereo
    images.
    
    timestamp: index used to choose the 
               stereo images pair.
    x0: x coordinate of the particle.
    y0: y coordinate of the particle
    yaw: Orientation of the particle.           

    '''
     
    
    path_l = os.path.join(path_left_stereo,stereo_images_left[timestamp])
    path_r = os.path.join(path_right_stereo,stereo_images_right[timestamp])
    
    image_l,disparity = pr2_utils.compute_stereo_texture(path_l, path_r)
    
    image_l = cv2.cvtColor(image_l,cv2.COLOR_BGR2RGB)
    
    K = np.array([[ 8.1690378992770002e+02, 5.0510166700000003e-01,
                   6.0850726281690004e+02], [0., 8.1156803828490001e+02,
                  2.6347599764440002e+02], [0., 0., 1. ]])
                                 
    fsu = K[0,0]
    
    baseline = 475.143600050775 * pow(10,-3)
    
    
    
    disparity = np.where(disparity==0,np.inf,disparity)
    depth = fsu * baseline/(disparity)
    
    z_l = (np.argwhere(depth>0)).T
    
    
    s_R_v = np.array([[-0.00680499,-0.0153215,0.99985], 
                      [-0.999977, 0.000334627,-0.00680066],
                      [-0.000230383,-0.999883,-0.0153234]])
    
    s_p_v = np.array([1.64239,0.247401,1.58411]).reshape(3,1)
    
    w_R_v = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    
    k_pi = K
    
    
    
    z_l_homo = homogenize(z_l) 
    
    m_stereo = np.linalg.inv(k_pi) @ z_l_homo 
    
    m_stereo = m_stereo * depth[z_l[0,:],z_l[1,:]]
    
    
    m_vehicle = s_R_v @ m_stereo + s_p_v
    
    m_world = w_R_v @ m_vehicle[0:2,:] + np.vstack((x0,y0)).reshape(2,1)
    
    
    x_cell = np.ceil((m_world[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_cell = np.ceil((m_world[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    x_cell[x_cell>=MAP['sizex']] = 0
    y_cell[y_cell>=MAP['sizey']] = 0
    x_cell[x_cell<0] = 0
    y_cell[y_cell<0] = 0
    MAP['texture_map'][x_cell,y_cell] = image_l[z_l[0],z_l[1]]

    
    
    

    
def update(time_stamp,lidar_data,angles,mu_part,alpha_old):
    '''
    
    Function to update map
    the map with lidar data.
    The probabilities of the 
    particles are also updated
    using the lidar data.
    
    time_stamp: Index to choose lidar
                data.
    lidar_data: Data from lidar scan
    angles: Angles covered by the 
            lidar scan.
    mu_part: Array of particles that
            contain the position
            and orientation of the 
            particles.
    alpha_old: Probabilities of
               the particles. 

    '''
    x0,y0 = mu_part[0,:],mu_part[1,:]
    
    yaw = mu_part[2,:]
    
    ranges = lidar_data[time_stamp, :]

    # take valid indices
    indValid = np.logical_and((ranges < 80),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    
    
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    
    # convert position in the map frame here 
    Y = np.stack((xs0,ys0))
    
    
    Y_homo = homogenize(Y)

    R = np.array([[0.00130201,0.796097,0.605167],[0.999999,-0.000419027,-0.00160026],[-0.00102038,0.605169,-0.796097]])
    
    p = np.array([[0.8349,-0.0126869,1.76416]]).T
    
    Y_homo_l2v = np.linalg.inv(R) @ (Y_homo - p)
    
    num_particles = x0.shape[0]
    
    
    Y_dehomo_l2v = Y_homo_l2v[:2,:]
    
    particle_coords = (np.vstack((x0,y0))).reshape(2,num_particles,1)

    
    w_R_v = np.zeros((num_particles,2,2))
    
    w_R_v[:,0,0] = np.cos(yaw)
    w_R_v[:,0,1] = -np.sin(yaw)
    w_R_v[:,1,0] = np.sin(yaw)
    w_R_v[:,1,1] = np.cos(yaw)
    
    Y_dehomo_l2v = w_R_v @ Y_dehomo_l2v
    
    Y_dehomo_l2v = np.swapaxes(Y_dehomo_l2v, 0, 1)
    
    Y_l2v_new = Y_dehomo_l2v + particle_coords
    
    
    
    x_coor_new = Y_l2v_new[0,:,:]
    y_coor_new = Y_l2v_new[1,:,:]
    
    xcell_new = np.ceil((x_coor_new - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    ycell_new = np.ceil((y_coor_new - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    xcell_new[xcell_new>=MAP['sizex']] = 0
    ycell_new[ycell_new>=MAP['sizey']] = 0
    
    xcell_new[xcell_new<0] = 0
    ycell_new[ycell_new<0] = 0
    
    matrix_map = MAP['map'][xcell_new,ycell_new]
    
    corr_new = np.sum(matrix_map,axis=1)
    
    
    
    obs_model = corr_new
    alpha_new = alpha_old * obs_model
    
    
    alpha_new = alpha_new/np.sum(alpha_new)
    
    max_prob_index = np.argmax(alpha_new)
    
    begin_x,begin_y = x0[max_prob_index],y0[max_prob_index]
    
    beginx_new = np.ceil((begin_x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    beginy_new = np.ceil((begin_y - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    
    ex,ey = xcell_new[max_prob_index,:],ycell_new[max_prob_index,:]
    
    MAP['log_map'][ex,ey]+=np.log(9)
    
    MAP['log_map']  = np.where(MAP['log_map']>50,50,MAP['log_map'])
    
    
    for i in range(Y_l2v_new.shape[2]):
        ex = xcell_new[max_prob_index,i]
        ey = ycell_new[max_prob_index,i]
        
        bresenham_coords = bresenham2D(beginx_new, beginy_new, ex, ey)
        
        x_coords,y_coords = bresenham_coords[0,1:-1],bresenham_coords[1,1:-1]
        
        x_coords = x_coords[np.argwhere(x_coords<MAP['sizex'])] 
        y_coords = y_coords[np.argwhere(y_coords<MAP['sizey'])] 
        MAP['log_map'][x_coords.astype(int),y_coords.astype(int)]-=np.log(9)
      
    MAP['log_map']  = np.where(MAP['log_map']<-50,-50,MAP['log_map'])
    MAP['map'] = np.exp(MAP['log_map'])/(1+np.exp(MAP['log_map']))
    
    if time_stamp%100 == 0:
        
        texture_map((time_stamp//100)-1, x0[max_prob_index], y0[max_prob_index],yaw[max_prob_index])
    


    return alpha_new     
    
    

def prediction(time_encoder,encoder_data,time_gyro,gyro_data,time_lidar,lidar_data,angles):
    '''
    Function for only prediction
    step.
    
    time_encoder: Time stamps of encoder
    encoder_data: Data from encoder.
    time_gyro: Time stamps of gyroscope.
    gyro_data: Data from gyroscope.
    time_lidar: Time stamps of lidar data.
    lidar_data: Data from Lidar scan.
    angles: Angles covered by the lidar scan.

    '''
    
    left_wheel_d = 0.623479
    right_wheel_d = 0.622806
    res = 4096
    
    len_lidar = len(time_lidar)
    
    num_particles = 300
    
    mu_part = np.zeros((3,num_particles))
    alpha = (1/num_particles) * np.ones(num_particles)
    

    cov = np.diag([0.1,0.001])
    
    threshold = num_particles * 0.2
    x_coor = []
    y_coor = []
    for i in range(1,len_lidar):
        
        prev_time = time_encoder[i-1]
        curr_time = time_encoder[i]
        
        t_encoder = curr_time - prev_time
        prev_ticks_left = encoder_data[i-1,0]
        curr_ticks_left = encoder_data[i,0]
        
        z_left = curr_ticks_left - prev_ticks_left
        
        prev_ticks_right = encoder_data[i-1,1]
        curr_ticks_right = encoder_data[i,1]
        
        z_right = curr_ticks_right - prev_ticks_right
        
        v_left = np.pi*left_wheel_d*z_left/(t_encoder*res)
        
        v_right = np.pi*right_wheel_d*z_right/(t_encoder*res)
        
        
        v = (v_left + v_right)/2
        
        
        gyro_indices = np.where((time_gyro>=prev_time) & (time_gyro<=curr_time))
        
        diff_theta = np.sum(gyro_data[gyro_indices,-1])
        
        
        t_gyro = time_gyro[gyro_indices[0][-1]] - time_gyro[gyro_indices[0][0]]
        
        omega = diff_theta/t_gyro
        
        noise = (np.random.multivariate_normal([0]*2,cov,num_particles))*pow(10,-9)
        

        v_new = v + noise[:,0]
        omega_new = omega + noise[:,1]
        
    
        mu_part[0,:] = mu_part[0,:] + t_gyro*(v_new*np.cos(mu_part[2,:]))
        mu_part[1,:] = mu_part[1,:] + t_gyro*(v_new*np.sin(mu_part[2,:]))
        mu_part[2,:] = mu_part[2,:] + t_gyro*omega_new
        
        
        if i%5==0:
            alpha = update(i,lidar_data,angles,mu_part,alpha)
            
            index = np.argmax(alpha)
            x_coor.append(mu_part[0,index])
            y_coor.append(mu_part[1,index])
            
            
            neff = 1/(np.sum(alpha*alpha))
            
            if neff<threshold:
                
                index = np.random.choice(num_particles,num_particles,p = alpha)
                
                mu_part = mu_part[:,index]
                
                alpha = (1/num_particles)*np.ones(num_particles)
        
        
        

    
    
    fig1,ax = plt.subplots(figsize=(10,8))
    x_coor = np.array(x_coor)
    y_coor = np.array(y_coor)
    xcell = np.ceil((x_coor - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    ycell = np.ceil((y_coor - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    MAP['map'][MAP['map']>0.5] = 1
    MAP['map'][MAP['map']<=0.5] = 0
    ax.imshow(MAP['map'],cmap='hot')
    ax.plot(ycell,xcell,'g',linewidth = 2)
    plt.title("Occupancy grid map")
    plt.show()
        
       
 
def dead_reckoning(time_encoder,encoder_data,time_gyro,gyro_data):
    '''
    The function implements the 
    dead reckoning step i.e
    implement only the prediction
    step for one particle
    without adding noise.


    time_encoder : Time stamp data from encoder
    encoder_data : Data from encoder
    time_gyro : Time stamps of gyroscope data
    gyro_data : Data from gyroscope.


    '''    
    
    left_wheel_d = 0.623479
    right_wheel_d = 0.622806
    res = 4096
    
    len_encoder = len(time_encoder)
    
    
    x = np.zeros((3,len_encoder))
    
    for i in range(1,len_encoder):
        
        prev_time = time_encoder[i-1]
        curr_time = time_encoder[i]
        
        t_encoder = curr_time - prev_time
        prev_ticks_left = encoder_data[i-1,0]
        curr_ticks_left = encoder_data[i,0]
        
        z_left = curr_ticks_left - prev_ticks_left
        
        prev_ticks_right = encoder_data[i-1,1]
        curr_ticks_right = encoder_data[i,1]
        
        z_right = curr_ticks_right - prev_ticks_right
        
        v_left = np.pi*left_wheel_d*z_left/(t_encoder*res)
        
        v_right = np.pi*right_wheel_d*z_right/(t_encoder*res)
        
        
        v = (v_left + v_right)/2
        
        
        gyro_indices = np.where((time_gyro>=prev_time) & (time_gyro<=curr_time))
        
        diff_theta = np.sum(gyro_data[gyro_indices,-1])
        
        
        t_gyro = time_gyro[gyro_indices[0][-1]] - time_gyro[gyro_indices[0][0]]
        
        omega = diff_theta/t_gyro
        

        x[:,i] = x[:,i-1] + t_gyro*np.array([[v*np.cos(x[2,i-1]),v*np.sin(x[2,i-1]),omega]]).reshape(3,1)
        
        
        
        

    
    
    x_coor = np.array(x[0,:])
    y_coor = np.array(x[1,:])

    plt.plot(x_coor,y_coor)
    plt.title("Dead Reckoning")
    plt.show()
        
if __name__=='__main__':
    
    path='data/sensor_data'
    
    encoder_path = os.path.join(path,'encoder.csv')
    gyro_path = os.path.join(path,'fog.csv')
    
    lidar_path = os.path.join(path,'lidar.csv')
    
    MAP = {}
    MAP['res']   = 1 #meters
    MAP['xmin']  = -150  #meters
    MAP['ymin']  = -1250
    MAP['xmax']  =  1250
    MAP['ymax']  =  150 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
    
    MAP['log_map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64)
    MAP['texture_map'] = np.zeros((MAP['sizex'],MAP['sizey'],3),dtype = np.int16)
    
    
    time_encoder,encoder_data = read_data_from_csv(encoder_path)
    time_gyro,gyro_data = read_data_from_csv(gyro_path)
    
    time_lidar, lidar_data = read_data_from_csv(lidar_path)
    angles = np.linspace(-5, 185, 286) / 180 * np.pi
    
    init_map(MAP, lidar_data, angles)
    
    pr2_utils.show_lidar() # to visualize lidar scan.
    
    dead_reckoning(time_encoder, encoder_data, time_gyro, gyro_data)
    
    path_left_stereo = 'data/stereo_images/stereo_left'
    path_right_stereo = 'data/stereo_images/stereo_right'
    
    stereo_images_left = sorted(os.listdir(path_left_stereo))
    stereo_images_right = sorted(os.listdir(path_right_stereo))
    
    
    start = pr2_utils.tic()
    prediction(time_encoder,encoder_data,time_gyro,gyro_data,time_lidar,lidar_data,angles)
    pr2_utils.toc(start)
    plot_texture_map(MAP)
    