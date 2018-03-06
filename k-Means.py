# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:55:54 2018

@author: pc
"""
import numpy as np

# class kmeans
class KMeans_Mike:
    def __init__(self,k=3,max_iter=100,rand=1,n_init=10,tol=1e-4,verbose=False,\
                 initial=[],seed=[],fix=False):
        '''
        centers is centers of cluster, dim = k * m
        Distance is distance to each of k cluster centers, dim = n * k
        labels is cluster grouping , dim = n * 1
        sse = sse of sampels to their closest cluster center **
        initial = [n_cluster,n_feature] centers (parameters)
        seed = {0:[row1, row2],1:[row8,row12]...}
        '''
        ##### first convert initial to numpy array check if initial & seed #####
        if (type(initial).__module__ != np.__name__):
            initial = np.array(initial)
        
        if initial.shape[0] > 0:
            if len(np.where(np.any(~np.isnan(initial),1))[0]) != len(seed):
            #if initial.shape[0] != len(seed):
                raise Exception('Initial clusters do not match number of seeds given')
            if initial.shape[0] > k:
                raise Exception('Initial clusters exceed number of clusters wanted')
        
        # set up attributes
        self.k = k
        self.max_iter = max_iter
        self.rand = rand
        if len(seed) < self.k:
            self.n_init = n_init 
        else: self.n_init = 1 # only need one iteration if all initial center is provided
        self.tol = tol 
        self.verbose = verbose
        self.fix = fix
        # here is to store all
        self.centers = []
        self.Distance = []
        self.labels = []
        self.inter_sse = np.zeros((n_init,1)) 
        # here is for in each initialization
        if initial.shape[0] > 0:
            self.c = initial
            self.seed = seed
        else:
            self.c = None
            self.seed = None
        self.d = None
        self.y = None
        self.sse = None    
        self.id = None
    
    def fit(self,X):
        '''
        X is data, dim = n * m 
        '''       
        for j in range(self.n_init):
            if self.c is None:
                # initialize
                np.random.seed(self.rand)
                self.c = X[np.random.choice(X.shape[0],self.k,replace=False),:] # any 4 points in X      
            else:
                missing = np.where(np.any(np.isnan(self.c),1))[0] # find rows with nan in clusters
                np.random.seed(self.rand)
                self.c[missing,:] = X[np.random.choice(X.shape[0],len(missing),replace=False),:]
                #for m in missing:
                #    np.random.seed(self.rand)
                #    self.c[m,:] = X[np.random.choice(X.shape[0],1,replace=False),:] # replace nan rows

            # if j == 0: print(self.c) # testing
            
            # initial distance and assign points into clusters
            self.dist(X) 
            self.findGroup() 
            (temp,self.sse,i) = (self.computeSSE(),10*self.computeSSE(),0) # temp stores sse to be changed
            if self.verbose: 
                print ("iter:{}, startsse:{}".format(j,temp))
                
            # start iterating
            while (i <self.max_iter and abs(self.sse-temp) > self.tol):
                self.sse = temp
                self.updateCenter(X) # self.y cannot be missing class
                self.dist(X)
                self.findGroup()
                temp = self.computeSSE()
                if (len(np.unique(self.y)) < self.k):  # if empty cluster 
                    # find the one with largest Distance and change cluster
                    self.assignNewCluster(X)
                if self.verbose: 
                    print('round: {}'.format(i))
                    print('previous: {}, new: {}, decrease: {}, group:{}'\
                            .format(self.sse,temp,self.sse-temp,np.unique(self.y)))
                i += 1
                
            # store results
            self.centers.append(self.c)
            self.Distance.append(self.d)
            self.labels.append(self.y)
            self.inter_sse[j] = self.sse
            self.rand+=1
            
        # compare results, return the best one
        ind = np.argmax(self.inter_sse)
        self.centers = self.centers[ind]
        self.Distance = self.Distance[ind]
        self.labels = self.labels[ind]
        self.inter_sse = self.inter_sse[ind]
    
    # steps in a cycle 
    def updateCenter(self,X): # returns new centers (k * m)
        for i in range(self.k):
            self.c[i,:] = np.mean(X[self.y==i,:],axis=0)           
    
    # steps in a cycle - 2    
    def dist(self,X): 
        # https://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html
        # distance
        d = self.c.T.reshape(1,X.shape[1],self.k) - np.repeat(X[:,:,np.newaxis],self.k,axis=2)
        # element-wise square, sum along row, e-w sqrt
        d = d**2 
        self.d = d.sum(1)**0.5 
    
    # steps in a cycle - 3 
    # v2. update this to not update the list of 
    def findGroup(self): #assign group, return Y (n,)
        self.y = np.argmin(self.d, axis=1)    
        if self.fix:
            for center in self.seed.keys():
                self.y[self.seed[center]] = center

    # use when empty cluster is encountered
    def assignNewCluster(self,X):
        # find which centers are missing
        ind = np.setdiff1d(np.arange(self.k),np.unique(self.y))
        print('missing labels: {}'.format(ind))
        for i in range(len(ind)):
            # find the index of sample that is the furthest away from all cluster centers
            temp = np.argmax(np.sum(self.d**2)) 
            self.y[temp] = ind[i] #change label
            self.dist(X)
        
    # math
    def computeSSE(self):
        temp = 0
        for i in range(self.k):
            temp += np.sum(self.d[self.y==i,i]**2)
        return temp