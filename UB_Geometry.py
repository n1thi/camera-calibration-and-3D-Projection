import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
Please do NOT read or write any file, or show any images in your final submission! 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    a = np.radians(alpha)
    b = np.radians(beta)
    g = np.radians(gamma)

    Rz_alpha = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0, 0, 1]
    ])

    Rx_beta = np.array([
        [1, 0, 0],
        [0, np.cos(b), -np.sin(b)],
        [0, np.sin(b),  np.cos(b)]
    ])

    Rz_gamma = np.array([
        [np.cos(g), -np.sin(g), 0],
        [np.sin(g),  np.cos(g), 0],
        [0, 0, 1]
    ])

    rot_xyz2XYZ = Rz_gamma @ Rx_beta @ Rz_alpha

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation
    R = findRot_xyz2XYZ(alpha, beta, gamma)
    rot_XYZ2xyz = R.T  # Transpose of rotation matrix is its inverse
    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1






#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    img_coord = np.array([
        
        [807, 645], [852, 660], [899, 675], [947, 691], 
        [1050, 692], [1103, 677], [1153, 662], [1201, 648], 
        [808, 711], [853, 727], [900, 744], [947, 761], 
        [1050, 762], [1102, 745], [1152, 730], [1200, 715], 
        [810, 777], [853, 794], [900, 811], [948, 829], 
        [1050, 830], [1102, 814], [1151, 797], [1199, 781], 
        [811, 841], [855, 859], [901, 878], [949, 898], 
        [1050, 898], [1101, 880], [1151, 862], [1198, 845]

        ])


    return img_coord

    



def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    world_coord = np.array([
        [0, 40, 40], [0, 40, 30], [0, 40, 20], [0, 40, 10], 
        [40, 0, 10], [40, 0, 20], [40, 0, 30], [40, 0, 40], 
        [0, 30, 40], [0, 30, 30], [0, 30, 20], [0, 30, 10], 
        [30, 0, 10], [30, 0, 20], [30, 0, 30], [30, 0, 40], 
        [0, 20, 40], [0, 20, 30], [0, 20, 20], [0, 20, 10], 
        [20, 0, 10], [20, 0, 20], [20, 0, 30], [20, 0, 40], 
        [0, 10, 40], [0, 10, 30], [0, 10, 20], [0, 10, 10], 
        [10, 0, 10], [10, 0, 20], [10, 0, 30], [10, 0, 40]
    ])

    return world_coord

def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''
    img_coord, world_coord = np.array(img_coord), np.array(world_coord)
    A = []
    for (X, Y, Z), (u, v) in zip(world_coord, img_coord):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    
    _, _, Vt = np.linalg.svd(np.array(A))
    P = Vt[-1].reshape(3, 4)  
    M, m1, m2, m3 = P[:, :3], P[0, :3], P[1, :3], P[2, :3]
    
    K = np.linalg.inv(np.linalg.qr(np.linalg.inv(M))[1])
    K /= K[2, 2]
    
    cx, cy = np.dot(m1, m3), np.dot(m2, m3)
    fx, fy = np.sqrt(np.dot(m1, m1) - cx*2), np.sqrt(np.dot(m2, m2) - cy*2)
    
    return fx, fy, cx, cy



def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''


    # Your implementation
    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)
    A = []
    for (X, Y, Z), (u, v) in zip(world_coord, img_coord):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    
    _, _, Vt = np.linalg.svd(np.array(A))
    P = Vt[-1].reshape(3, 4)
    M, t_ = P[:, :3], P[:, 3]
    
    Q, R_inv = np.linalg.qr(np.linalg.inv(M))
    R, t = np.linalg.inv(Q), np.dot(np.linalg.inv(R_inv), t_.reshape(3, 1))
    
    return R, t.flatten()


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2

def transpose_matrix(mat):
    return [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]

def matmul(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def normalize(vec):
    norm = sum(x*x for x in vec) ** 0.5
    return [x / norm for x in vec]

def power_iteration_smallest_eigenvector(mat, num_iter=1000, epsilon=1e-10):
    n = len(mat)
    v = [1.0] * n
    v = normalize(v)

    lambda_max = 0
    for _ in range(num_iter):
        # Inverse iteration idea: (A - mu*I)^-1
        Av = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
        lambda_max = sum(Av[i] * v[i] for i in range(n))
        v = normalize(Av)

    return v  # Approximate smallest eigenvector





#---------------------------------------------------------------------------------------------------------------------