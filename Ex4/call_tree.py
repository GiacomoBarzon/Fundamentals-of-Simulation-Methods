from tree import *
from random import *
import time
from copy import *
from math import *
import matplotlib.pyplot as plt


def main(N, anglemax, part, f_exact, show = False, exact = False):
    #
    # Create a set of randomly positioned particles
    # For convenience we assume all masses to be 1.
    # If we have in reality another mass, we can simply
    # rescale our answers.
    #
    
    nparticles = N
    particles = []
    for i in range(nparticles):
        x = random()
        y = random()
        z = random()
        particles.append([x,y,z])

    particles = np.array(particles)
    
    
    if (exact == True):
        
        print("\n Starting N^2 gravity of " + str(N) + " particles: \n")
    
        m = 1./nparticles             # potential constants
        epsilon = 0.001

        force_exact = np.zeros((nparticles, nparticles, 3))    # consider interaction of a particle with all the others

        t0 = time.time()

        for i in range(0, nparticles):          # here fill only the superior half
            for j in range(i+1, nparticles):    # start at i+1, avoid self interaction
                force_exact[i,j,:] = m**2 * (particles[i,:]-particles[j,:]) / ((np.sum((particles[i,:]-particles[j,:])**2) + epsilon**2)**1.5)


        force_exact = force_exact - np.transpose(force_exact, (1,0,2))    # create entire matrix
        #print(force_exact[:,:,0])
        mod_force_exact = np.sqrt(np.sum((np.sum(force_exact, axis=1))**2, axis = 1))   # calculate modulus of force on the particles


        t1 = time.time()
        fullgrav_dt = t1-t0
        print("Done in "+str(fullgrav_dt)+" seconds\n")
        

        #
        # Now create the tree
        #
        q = TreeClass(particles)
        q.insertallparticles()
        q.computemultipoles(nodeid = 0)

        #
        # Compute forces 
        #
        print("Starting tree gravity of " + str(N) + " particles with theta = " + str(anglemax) + " : \n")
        t0 = time.time()

        q.allgforces(anglemax)

        t1 = time.time()
        treegrav_dt = t1-t0
        print("Done in "+str(treegrav_dt)+" seconds\n")

        fapprox = deepcopy(q.forces)
        N_int = deepcopy(q.N_nodes)

        mod_fapprox = np.sqrt(np.sum(fapprox**2, axis=1))
        
        #for i in range(len(mod_fapprox)):
         #   print(fapprox[i,:], ', ', np.sum(force_exact, axis=0)[i,:], '\n')
         #   print(mod_fapprox[i], ', ', mod_force_exact[i], ' ', N_int[i], '\n') 
         #   print('\n -------------------------------------------------------')

        if (show == True):
            print('Number of nodes: ', N_int)
            
        print('\n Mean number of interactions: ', np.mean(N_int), ' +- ', np.std(N_int))


        if (show == True):
            print('f_approx - f_exact: \n', mod_fapprox - mod_force_exact)

        # 
        # Now compare the approximate and exact versions
        #
        eta = np.zeros((nparticles))
        #eta = abs(mod_fapprox - mod_force_exact) / mod_force_exact
        eta = np.sqrt(np.sum((fapprox - np.sum(force_exact, axis=0))**2, axis=1) ) / mod_force_exact
        print('\n Mean error with tree algorithm: ', np.mean(eta), ' +- ', np.std(eta))
        print('\n -------------------------------------------------------')


        if (show == True):
            fig = plt.figure(figsize=[12,7])
            histo, bins, pathches = plt.hist(eta, 50, density = True)
            plt.xlabel('$\eta$', fontsize = 15)
            plt.ylabel(' ', fontsize = 15)
            plt.title('Normalized Histogram of errors', fontsize = 15)
            plt.show(fig)
        
        return treegrav_dt, fullgrav_dt, eta, N_int, particles, mod_force_exact
            
    else:
        #
        # Now create the tree
        #
        
        particles = part
        
        q = TreeClass(particles)
        q.insertallparticles()
        q.computemultipoles(nodeid = 0)

        #
        # Compute forces 
        #
        print("Starting tree gravity of " + str(N) + " particles with theta = " + str(anglemax) + " : \n")
        t0 = time.time()

        q.allgforces(anglemax)

        t1 = time.time()
        treegrav_dt = t1-t0
        print("Done in "+str(treegrav_dt)+" seconds\n")

        fapprox = deepcopy(q.forces)
        N_int = deepcopy(q.N_nodes)

        mod_fapprox = np.sqrt(np.sum(fapprox**2, axis=1))

        if (show == True):
            print('Number of nodes: ', N_int)
            
        print('\n Mean number of interactions: ', np.mean(N_int), ' +- ', np.std(N_int))


        if (show == True):
            print('f_approx - f_exact: \n', mod_fapprox - f_exact)

        # 
        # Now compare the approximate and exact versions
        #
        eta = np.zeros((nparticles))
        eta = abs( mod_fapprox - f_exact ) / f_exact
        print('\n Mean error with tree algorithm: ', np.mean(eta), ' +- ', np.std(eta))
        print('\n -------------------------------------------------------')


        if (show == True):
            fig = plt.figure(figsize=[12,7])
            histo, bins, pathches = plt.hist(eta, 50, density = True)
            plt.xlabel('$\eta$', fontsize = 15)
            plt.ylabel(' ', fontsize = 15)
            plt.title('Normalized Histogram of errors', fontsize = 15)
            plt.show(fig)

    
        return treegrav_dt, eta, N_int
