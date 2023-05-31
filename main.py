import os
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#This library is used to compute euler angles from rotation matrix
from scipy.spatial.transform import Rotation as R

from hough import *


#-----------------------------------------------
#Problem1. Question 1
#-----------------------------------------------

#Class provides functionality to plot multiple plots
class signal_plotter():
    def prepare_plot(self, plot_identifier=11, title=""):
        '''
        This function prepares matplot for plotting signal data
        :param plot_identifier: describes nuimber of subplots
        :param title: describes main title of the plot
        '''
        self.fig = plt.figure()
        # self.fig.canvas.set_window_title(title)
        self.fig.suptitle(title, fontsize=30)
        self.plot_identifier = plot_identifier

    def make_subplot(self, plot_num  = 1, title="", x_label="", y_label="", ):
        '''
        This function creates a subplot for plotting data
        :param plot_num: subplot number in the whole grid
        :param title: tilte for the subplot
        :param x_label: x_label for the subplot
        :param y_label: y_label for the subplot
        '''
        sub_plot =  self.fig.add_subplot(self.plot_identifier*10+plot_num)
        sub_plot.title.set_text(title)
        sub_plot.xaxis.set_label_text(x_label)
        sub_plot.yaxis.set_label_text(y_label)
        return sub_plot

    def plot_signal(self, signal, signal_name="", sub_plot = None):
        '''
        This function plots a signal in the given subplot
        :param plot_num: subplot number in the whole grid
        :param signal: data to be plotted
        :param signal_name: name of the signal
        :param sub_plot: subplot instance in which the data needs to be plotted
        '''
        sub_plot.plot(signal, label=signal_name, linewidth=2)
        sub_plot.legend(loc='upper left', frameon=False)

    def show_plot(self):
        '''This funciton displys the final plot'''
        plt.show()




video_file = os.path.join(os.path.dirname(__file__), "data", "video.avi")
clip = cv.VideoCapture(video_file)

camera_pose_x = []
camera_pose_y = []
camera_pose_z = []
camera_pose_alpha = []
camera_pose_beta = []
camera_pose_gamma = []

if clip.isOpened():
    width  = clip.get(3)   
    height = clip.get(4) 

    while(True):
        #read a frame from the video
        _ , original_frame = clip.read()

        if original_frame is not None:
            frame  = original_frame.copy()
            
            if frame is None:
                clip.release()
                cv.destroyAllWindows()
                break
            #----------------------------------------
            #create filter mask to track the ball
            #----------------------------------------

            #convert frame to HSV
            frame_hsv  = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            color_mask = cv.inRange(frame_hsv, (82,15,242), (185, 255, 255))

            #Remove noise--------------------------------------------------------
            avg_kernel = np.ones((5,5))
            color_mask = cv.morphologyEx(color_mask, cv.MORPH_CLOSE, avg_kernel)
            #--------------------------------------------------------------------

            #Detect edges--------------------------------------------------------
            color_mask = cv.Canny(color_mask, 300, 400)
            #--------------------------------------------------------------------
            
            #Compute hough lines-------------------------------------------------
            _hough_compute = hough_compute()
            lines, rho_bins, angle_bins, parallel_set1, parallel_set2 = _hough_compute.get_hough_lines(color_mask, width, height)
            #--------------------------------------------------------------------
            
            #Compute corners-------------------------------------------------
            corners = _hough_compute.find_corners( parallel_set1, parallel_set2, rho_bins, angle_bins )
            #--------------------------------------------------------------------
           

            #If valid number of corners are found-----------------------------
            if len(corners) == 4:
                point1 = corners[0]
                point2 = corners[1]
                point3 = corners[2]
                point4 = corners[3]
                
                #Draw border lines around the paper---------------------------------
                cv.line(frame, point1, point3, (0,255,255), 2)
                cv.line(frame, point4, point2, (0,255,255), 2)
                cv.line(frame, point4, point3, (0,255,255), 2)
                cv.line(frame, point1, point2, (0,255,255), 2)
                #--------------------------------------------------------------------


                #Draw dots near the corners of the paper---------------------------------
                cv.circle(frame, point1, 10, (255,0,0), -1)
                cv.circle(frame, point2, 10, (255,0,0), -1)
                cv.circle(frame, point3, 10, (255,0,0), -1)
                cv.circle(frame, point4, 10, (255,0,0), -1)
                #--------------------------------------------------------------------


                #Compute world origin location at the middle of the paper------------
                World_origin = np.array((point1+point4)/2).astype(np.int)
                axis_y = np.array((point2+point4)/2).astype(np.int)
                axis_x = np.array((point3+point4)/2).astype(np.int)
                #--------------------------------------------------------------------
                

                #Draw origin and axis-----------------------------------------------
                cv.circle(frame, World_origin, 10, (0,0,255), -1)
                cv.arrowedLine(frame, World_origin, axis_x, (0,0,255), 2)
                cv.arrowedLine(frame, World_origin, axis_y, (0,0,255), 2)
                #--------------------------------------------------------------------

                #Given camera matrix with no resize----------------------------------
                K = np.array([[1.38E+03,	0,	9.46E+02],
                            [0,	1.38E+03,	5.27E+02],
                            [ 0,	0,	1]])
                #--------------------------------------------------------------------
                

                #Compute location of the the corners in the world coordinate system--
                paper_height = 21.6
                paper_width = 27.9
                px = paper_width/2
                py = paper_height/2

                real_world_point1 = [-px, -py]
                real_world_point2 = [-px,  py]
                real_world_point3 = [ px, -py]
                real_world_point4 = [ px,  py]
                #--------------------------------------------------------------------

                # We know that 
                #    Image_point = gamma * K * R * world_point
                #    Where [gamma * K * R] is called the projection matrix P
                real_world_points = [real_world_point1, real_world_point2, real_world_point3, real_world_point4]
                Image_points = [point1, point2, point3, point4]
                
                #create linear equations to solve for projection matrix
                A = []
                for i in range(len(Image_points)):
                    x1,y1 = real_world_points[i]
                    p1,q1 = Image_points[i]
                    A.append([-x1, -y1, -1, 0, 0, 0, x1*p1, y1*p1, p1])
                    A.append([0, 0, 0, -x1, -y1, -1, x1*q1, y1*q1, q1])
                #Solve for linear equations using SVD
                U, S, Vt = np.linalg.svd(A)
                #dive with last element of the matrix to compensate for the 4 point solution
                H = Vt[-1, :] / Vt[-1, -1]
                H = H.reshape(3, 3)

                #Compute [gamma * R] from homography matrix
                camera_pose = np.linalg.inv(K).dot(H)

                #Calcualte the value of gamma from Rx and Ry columns
                gamma1 = np.linalg.norm(H[:,0])
                gamma2 = np.linalg.norm(H[:,1])
                gamma = (gamma1+gamma2)/2
                #Compute the theird column of the rotation matrix
                Rz = np.cross(H[:,0], H[:,1])

                #Divide by stack columns and gamma to get the final rotaiton matrix
                Rot = np.hstack((H[:,0].reshape(3,1), H[:,1].reshape(3,1), Rz.reshape(3,1)))/gamma
                
                #evaluate Rool, Pitch and Yaw from the rotation matrix
                r = R.from_matrix(Rot)
                roll, pitch, yaw = r.as_euler("xyz", degrees=True)

                #Record the 6DOF of the camera pose
                camera_pose_x.append(camera_pose[0][2])
                camera_pose_y.append(camera_pose[1][2])
                camera_pose_z.append(camera_pose[2][2])

                camera_pose_alpha.append(pitch)
                camera_pose_beta.append(roll)
                camera_pose_gamma.append(yaw)

            #Show the video
            cv.imshow("video" ,frame)

        else:
            clip.release()
            cv.destroyAllWindows()
            break
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            clip.release()
            cv.destroyAllWindows()
            break


if __name__ == "__main__":

    _signal_plotter = signal_plotter()
    _signal_plotter.prepare_plot(32, "camera pose plots")

    #plot the 6DOF camera pose values---------------------------------------------------------------
    subplot = _signal_plotter.make_subplot(1, "", "frame number", "x movement")
    _signal_plotter.plot_signal(camera_pose_x, "camera pose - x in cm", subplot)

    subplot = _signal_plotter.make_subplot(2, "", "frame number", "y movement")
    _signal_plotter.plot_signal(camera_pose_y, "camera pose - y in cm", subplot)

    subplot = _signal_plotter.make_subplot(3, "", "frame number", "z movement")
    _signal_plotter.plot_signal(camera_pose_z, "camera pose - z in cm", subplot)

    subplot = _signal_plotter.make_subplot(4, "", "frame number", "roll movement")
    _signal_plotter.plot_signal(camera_pose_alpha, "camera pose - roll in degrees", subplot)

    subplot = _signal_plotter.make_subplot(5, "", "frame number", "pitch movement")
    _signal_plotter.plot_signal(camera_pose_beta, "camera pose - pitch in degrees", subplot)

    subplot = _signal_plotter.make_subplot(6, "", "frame number", "yaw movement")
    _signal_plotter.plot_signal(camera_pose_gamma, "camera pose - yaw in degrees", subplot)

    _signal_plotter.show_plot()
    #------------------------------------------------------------------------------------------------
