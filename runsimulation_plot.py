# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:09:26 2014

@author: koher
"""
#%%
import pyximport
pyximport.install()
from lattice import update_lattice
import numpy as np
import numpy.random as rn
#%%

def runsimulation_plot(N_ensembles = 1,N_runs = 1000, dim = 50, p = 0.3, mu = 1./100, path = '/users/stud/koher/arbeit/'):
    N_ensembles = int(N_ensembles)
    N_runs = int(N_runs)
    dim = int(dim)
    p = float(p)
    mu = float(mu)
    path = str(path)
    
    maxdist = np.empty((1,N_ensembles),dtype=float)[0]
    dist = np.empty((1,N_runs),dtype=float)[0]
    for jj in xrange(N_ensembles):
        
        bonds = (rn.rand(2,dim,dim) < p).astype(int)
        times = rn.exponential(scale=1/mu, size=(2,dim,dim))
        pos  = np.array([int(dim/2),int(dim/2)])
        pos0 = np.array([int(dim/2),int(dim/2)])
        pos2 = np.array([dim/2+1,dim/2])
        system_time = 0
        
        vlist = [bonds[0].copy()]
        hlist = [bonds[1].copy()]
        plist = [pos.copy()]
        p2list= [pos2.copy()]
        
        for run in xrange(N_runs):
            x, y = pos[0], pos[1]
            neighbours = [[bonds[0,x,y],0],
                          [-bonds[0,x-1,y],0],
                          [0,-bonds[1,x,y-1]],
                          [0,bonds[1,x,y]]]
            choice = rn.randint(0,4)
            pos[0] += neighbours[choice][0]
            pos[1] += neighbours[choice][1]
            
            x, y = pos2[0], pos2[1]
            neighbours = [[int(bonds[0][x,y]),0],
                          [-int(bonds[0][x-1,y]),0],
                          [0,-int(bonds[1][x,y-1])],
                          [0,int(bonds[1][x,y])]]
            choice = rn.randint(0,4)
            pos2[0] += neighbours[choice][0]
            pos2[1] += neighbours[choice][1]
            
            system_time = update_lattice(bonds,times,system_time,p,mu,dim)
            newdist = (pos[0] - pos0[0])**2 + (pos[1] - pos0[1])**2
            dist[run] = newdist
            
            vlist.append(bonds[0].copy())
            hlist.append(bonds[1].copy())
            plist.append(tuple(pos.copy()))
            p2list.append(tuple(pos2.copy()))
            
            if newdist > maxdist[jj]:
                maxdist[jj] = newdist
            
            if (pos[0] == pos2[0]) and (pos[1] == pos2[1]):
                plist.append(np.array([0,0]))
                p2list.append(np.array([0,0]))
                vlist.append(bonds[0].copy())
                hlist.append(bonds[1].copy())
                print 'game over'
    
    dist = dist/N_ensembles
    #fname = path + 'p' + str(p) + '_mu' + str(mu)
    #D = np.polyfit(range(N_runs-1000,N_runs), dist[-1000::], 1)
    #np.savez(fname, dist=dist, p=p, mu=mu, N_runs=N_runs, N_ensembles=N_ensembles, dim=dim, D=D, maxdist=maxdist)
    return hlist, vlist, plist, p2list
#%%

if __name__ == "__main__":
    dim = 20
    p1 = 0.03
    p2 = 0.05
    p = p2/(p1+p2)
    mu = p1 + p2
    bonds = (rn.rand(2,dim,dim) < p).astype(int)
    times = rn.exponential(scale=1.0/mu, size=(2,dim,dim))
    system_time = 1
    N_runs = 100
    N_ensemble = 1
    
    hlist, vlist, pos, pos2 = runsimulation_plot(N_ensemble, N_runs, dim, p, mu, path='/users/stud/koher/arbeit/')
    
    #import matplotlib
    #matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    from matplotlib import animation
    
    
    #pos = [(int(p[0]),int(p[1])) for p in plist]
    #pos2 = [(int(p[0]),int(p[1])) for p in p2list]

    N = 20
    N_runs = len(pos)
    patches = []
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(5, 3.2)
    ax = plt.axes(xlim=(0, N), ylim=(0, N))
    ax.set_frame_on(False)
    
    for ii in range(2 * N**2):
        ax.plot([],[], 'k-', lw=2)
    lines = ax.lines
    cir = plt.Circle(pos[0], 0.3, fc='r')
    cir2 = plt.Circle(pos[0], 0.3, fc='b')
    #wed = Wedge((.5,.5), 0.3, 30, 320,fc='r')
    ax.add_patch(cir)
    ax.add_patch(cir2)
    tex = ax.text(int(N/2)-4,int(N/2), '',fontsize=20)
    #ax.add_patch(wed)
    
    def init():
        for line in patches[0:-2]:
            line.set_data([], [])
        cir.set_visible(False)
        cir2.set_visible(False)
        #wed.set_visible(False)
        tex.set_text('')
        return lines + [cir,cir2,tex]
    
    def animate(i):
        y,x = pos[i]
        y2,x2 = pos2[i]
        ind = 0
        if pos[i] != (0,0):
            for ii in range(N):
                for jj in range(N):
                    if vlist[i][ii,jj]:
                        lines[ind].set_data([jj,jj],[ii+1,ii])
                    elif not(vlist[i][ii,jj]):
                        lines[ind].set_data([],[])
                    ind += 1
                    if hlist[i][ii,jj]:
                        lines[ind].set_data([jj,jj+1],[ii,ii])
                    elif not(hlist[i][ii,jj]):
                        lines[ind].set_data([],[])
                    ind += 1
            
            cir.center = x,y
            cir.set_visible(True)
            
            cir2.center = x2,y2
            cir2.set_visible(True)
            tex.set_text('')
        else:
            tex.set_text('Game Over')
            ind = 0
            for ii in range(N):
                for jj in range(N):
                    lines[ind].set_data([],[])
                    ind += 1
                    lines[ind].set_data([],[])
                    ind += 1
            cir.set_visible(False)
            cir2.set_visible(False)
        
        '''
        if (i % 2):
            cir.center = x,y
            cir.set_visible(True)
            wed.set_visible(False)
        else:
            wed.center = x,y
            wed.set_visible(True)
            cir.set_visible(False)
        '''
        return lines + [cir,cir2,tex]
    
    anim = animation.FuncAnimation(fig, animate, 
                                   init_func=init, 
                                   frames=N_runs, 
                                   interval=200,
                                   repeat_delay=2000,
                                   blit=True)
    plt.show()
    