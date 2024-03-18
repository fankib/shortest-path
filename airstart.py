import numpy as np
import matplotlib.pyplot as plt

# helper

def angle(a, b):
    cos_ab = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.rad2deg(np.arccos(cos_ab))

def is_dominant(a, b, n):
    sign_ab = (b-a).dot(n)
    return sign_ab <= 0

def compute_normal(sss, wp1, ratio):    
    d = wp1 - sss
    degs = np.rad2deg(np.arctan(ratio))
    #print('use degs:', degs)
    cos_alpha = np.cos(np.deg2rad(degs))
    sin_alpha = np.sin(np.deg2rad(degs))
    #print('cos_alpha', cos_alpha)
    r = sin_alpha / np.linalg.norm(d)
    return np.array([r*d[0], r*d[1], cos_alpha])

if __name__ == '__main__':

    # Define Directio d:
    sss = np.array([3, 1, 0])
    wp1 = np.array([5, 2, 0])
    #wp1 = np.array([1, 0.5, 0])
    d = wp1 - sss    

    # Compute Normal of Domination Plane:
    ratio = 1/3 # Domination Glide Angle
    n = compute_normal(sss, wp1, ratio)

    print('n unit length?', np.linalg.norm(n))
    #print('d not unit length?', np.linalg.norm(d))
    # compute angles:
    print('deg(n, d\') = ', angle(n, np.array([0, 0, 1])))
    print('deg(n, d) = ', angle(n, d))

    # Draw Points and Normals:
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(sss[0], sss[1], sss[2], label='sss')
    ax.scatter(wp1[0], wp1[1], wp1[2], label='wp1')

    ax.plot([0, 0], [0, 0], [0, 1], label='d\'')
    ax.plot([0, d[0]], [0, d[1]], [0, d[2]], label='d')
    ax.plot([0, n[0]], [0, n[1]], [0, n[2]], label='n')

    # Define Pilots:
    pilot_A = np.array([1, 1, 3])
    pilot_B = np.array([2, 2, 2])

    ax.scatter(pilot_A[0], pilot_A[1], pilot_A[2], label='A')
    ax.scatter(pilot_B[0], pilot_B[1], pilot_B[2], label='B')
    print('Dominance A over B: (<0)', is_dominant(pilot_A, pilot_B, n))
    #print('Dominance B over A: (<0)', (pilot_A-pilot_B).dot(n))

    # plot
    ax.set_aspect('auto')

    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 8)
    ax.set_zlim(-2, 8)

    ax.legend()
    plt.show()
