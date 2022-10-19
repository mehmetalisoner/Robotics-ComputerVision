# Problem 2

import numpy as np
from numpy import linalg
import scipy as scp
from scipy.linalg import svd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import math

def createCameraMatrix(Rrow1, Rrow2, Rrow3, tvec, Krow1,Krow2,Krow3):
    R_W = np.array([Rrow1,Rrow2,Rrow3])
    t_W = np.array(tvec)
    t_W = t_W.reshape((-1,1))
    T_W = np.hstack([R_W,t_W])
    K = np.array([Krow1,Krow2,Krow3])
    M = np.dot(K,T_W)
    
    return M, K, T_W, R_W, t_W

def MdotPoints(points,matrix,output):
    for point in points:
        temp = np.dot(matrix,point)
        output.append(temp/(temp[2]))

def pixelTo2D(point,sigma_x,sigma_y,sx,sy,focal_length):
    newx = (point[0]-sigma_x) * (-sx)
    newy = (point[1] - sigma_y) * (-sy)
    return np.array([newx,newy,focal_length])

def createTwX(matrix):
    return np.vstack([matrix, [0,0,0,1]])


def solveRayEqs(pl,pr,RrL,trL):
    q = np.cross(pl, np.dot(RrL,pr))
    lenq = math.sqrt(pow(q[0],2) + pow(q[1],2) + pow(q[2],2))
    q = q/lenq
    alpha = - np.dot(RrL, pr)
    mtx = np.array([pl, alpha,q]).T
    # s, v, d = np.linalg.svd(mtx)
    a,b,c = np.linalg.solve(mtx,trL)
    return a,b,c,q

def calculatePW (pl,a,c,q,TwL):
    PL = a * pl + (c/2) * q
    PL = np.hstack([PL,[1]]).T
    print(PL)
    PW = np.dot(np.linalg.inv(TwL),PL)
    return PW, PL

# K matrix & its parameters
Krow1 = [-100,0,200]
Krow2 = [-0,-100,200]
Krow3 = [0,0,1]
sigma_x = Krow1[2]
sigma_y = Krow2[2]
focal_length = 100
sx = -focal_length/Krow1[0]
sy = -focal_length/Krow2[1]



# Parameters for Left Camera
R1_row1 = [0.707,0.707,0]
R1_row2 = [-0.707,0.707,0]
R1_row3 = [0,0,1]
t1_vec = [-3,-0.5,3]

M1,K,T1,R1,t1 = createCameraMatrix(R1_row1, R1_row2,R1_row3,t1_vec,Krow1,Krow2,Krow3)


# Parameters for Right Camera
R2_row1 = [0.866,-0.5,0]
R2_row2 = [0.5,0.866,0]
R2_row3 = [0,0,1]
t2_vec = [-3,-0.5,3]

M2,K,T2,R2,t2 = createCameraMatrix(R2_row1, R2_row2,R2_row3,t2_vec,Krow1,Krow2,Krow3)


# Creating a cube, defining the points in world-coord.
p0 = np.array([0,0,0,1])
p1 = np.array([1,0,0,1])
p2 = np.array([0,0,1,1])
p3 = np.array([1,0,1,1])
p4 = np.array([0,1,1,1])
p5 = np.array([1,1,1,1])
p6 = np.array([0,1,0,1])
p7 = np.array([1,1,0,1])

points_3d = [p0,p1,p2,p3,p4,p5,p6,p7]


# Going from 3d to 2d, to image coords.
pointsFromFirst  = []  # store all M1 * points_3D
pointsFromSecond = []  # store all M2 * points_3D

# Multiply every point in the cube with each matrix and store them
MdotPoints(points_3d,M1,pointsFromFirst)
MdotPoints(points_3d,M2,pointsFromSecond)


pl_0 = pixelTo2D(pointsFromFirst[0],sigma_x,sigma_y,sx,sy,focal_length)
pr_0 = pixelTo2D(pointsFromSecond[0],sigma_x,sigma_y,sx,sy,focal_length)



TwL = createTwX(T1)         # create T_w^L, for left camera
TwR = createTwX(T2)         # create T_w^R, for right camera
TrW = np.linalg.inv(TwR)    # invert T_w^R to get T_r^W


TrL = np.dot(TwL,TrW)       # combine both to get TrL
RrL = TrL[0:3,0:3]          # extract RrL
trL = TrL[0:3,3]            # extract trL

# s,v,d = solveRayEqs(pl_0,pr_0,RrL,trL)
# x = d*(diag(diag(v).^-1)*(s.'*b))
# print(s)
# print("\n",v.T)
# print("\n",d)

a,b,c,q = solveRayEqs(pl_0,pr_0,RrL,trL)

print(a,b,c,q)

pw,pl = calculatePW(pl_0,a,c,q,TwL)

print(pw)