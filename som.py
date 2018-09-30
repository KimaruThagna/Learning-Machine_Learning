import numpy as np
import matplotlib.pyplot as plt

N = 15 # linear size of 2D map
n_teacher = 1000 # # of teacher signal
np.random.seed(100)# test seed for random number

def kohonen():
    # initialize grid
    nodes = np.random.rand(N,N,3)#Each node has a 3d weight vector
    plt.imshow(nodes, interpolation='none')
    plt.savefig("init.png")
    # teacher signal
    teachers = np.random.rand(n_teacher,3)
    for i in range(n_teacher):
        train(nodes, teachers, i)
        plt.imshow(nodes, interpolation='none')
    plt.savefig("final.png")
    
def train(nodes, teachers, i):
    bmu = best_matching_unit(nodes, teachers[i]) # coords of the node closest to the teacher
    print(bmu)
    for x in range(N): # per row
        for y in range(N):# per column at row x
            c = np.array([x,y])# coordinate of unit
            d = np.linalg.norm(c-bmu) #euclidian dist
            L = learning_ratio(i)
            S = learning_radius(i,d)
            for z in range(3): # per value of node at row x,col y
                nodes[x,y,z] += L*S*(teachers[i,z] - nodes[x,y,z])# teacher-system...add delta(modified by the decay func


def best_matching_unit(nodes, teacher):
    #compute all squared errors
    norms = np.zeros((N,N)) # initialize to an NXN matrix of 0s
    for i in range(N):
        for j in range(N):
            for k in range(3):
                norms[i,j] += (nodes[i,j,k] - teacher[k])**2 #squared error
    #then, choose the minimum one
    # norms is an array(NXN) of squared errors,
    bmu = np.argmin(norms) #minimum element ie, least error
    return np.unravel_index(bmu,(N,N))
    # get position of bmu from the NXN grid returns a row by col index

def neighbourhood(t):#neighbourhood radius
    halflife = float(n_teacher/4)
    initial  = float(N/2)# half size of lattice
    return initial*np.exp(-t/halflife)

def learning_ratio(t):
    halflife = float(n_teacher/4)
    initial  = 0.99
    return initial*np.exp(-t/halflife) # decay the learning rate

def learning_radius(t, d):
    # d is distance from BMU
    s = neighbourhood(t)
    return np.exp(-d**2/(2*s**2)) # gaussian decay

kohonen()