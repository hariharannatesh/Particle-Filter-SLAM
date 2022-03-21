# Particle Filter SLAM
The main file used to run the Particle Filter slam is particle_filter_texture_map.py. Its uses functions from pr2_utils.py such as 
read_data_from_csv(), bresenham2D(), compute_stereo_texture().

The functions used in this python file are:

1. homogenize(): It converts a vector to homogenous form.
2. plot_map() : It plots the Occupancy grid map alone.
3. plot_texture_map(): It plots the texture map alone.
4. init_map(): It initializes the occupancy grid map.
5. texture_map(): It builds a texture map using the stereo images.
6. update(): It updates the probabilities of the particles and also updates the occupancy grid map. If the timestamp is divisible by 100,
             the texture_map() function is called.
7. prediction(): It predicts the states of the particles using the encoder data. If the time step is divisible by 5, the update() function
                is called.
8. dead_reckoning(): It does the prediction step of one particle without noise and plots the path.

To visualize the laser scan, pr2_utils.show_lidar() is called. The Laser scan at time t=0 is plotted in init_map().

Run the prediction() function by passing the necessary parameters as mentioned in the description of the functions. 

Click [here](https://drive.google.com/file/d/1N2-uOkKNDntxpfr1eshez0w-XrpX-3K7/view?usp=sharing) to view the report.

