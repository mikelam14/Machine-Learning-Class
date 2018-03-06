# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:57:16 2018

@author: pc
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import pickle
except ImportError:
    pickle = None

class mini_batch_GD:
    def __init__(self,alpha=0.1,bs=3,epoch=3,mu=0,threshold=0.5,testing=False,n_plot=1000):
        '''
        # X is n by m+1 (m is number of feature, 1 is bias)
        # y is n by 1
        # p is bs by 1 --> p means P(Y=1|X)
        # weight is m+1 by 1
        # gradient is m+1 by 1
        # alpha, batch_size, mu, epoch, threshold are scalars
        delta = d(error)/d_w 
        '''
        self.alpha = alpha
        self.batch_size = bs
        self.mu = mu
        self.epoch = epoch
        self.threshold = threshold
        self.testing = testing
        self.validation = False
        self.best_val = None
        self.n_plot = n_plot
        self.save = False
    
    def initialize(self,X): # called only at first for a model
        self.weights = np.zeros(X.shape[1]).reshape(-1,1)  # m+1 by 1
        self.gradients = np.zeros(X.shape[1]) # m+1 by 1
        if self.testing:
            self.dataSeen = 0
            self.plotPoint = 0
            self.error = [[],[]] # [[training], [testing error]]
            self.e_array = [[],[]]
        #self.p = np.zeros(X.shape[0]).reshape(-1,1)
     
    def predict(self,X,val=False): 
        # get p, X could be one point, batch or all. Can be used for both training and validation
        # (n-by-m+1 * m+1-by-1) or (bs-by-m+1 * m+1-by-1)
        if val:
            self.val_p = np.exp(np.dot(X,self.weights))/(1+np.exp(np.dot(X,self.weights)))    
        else:
            self.p = np.exp(np.dot(X,self.weights))/(1+np.exp(np.dot(X,self.weights)))
    ''' 
    seems redundant - TBC
    
    def prob(self,X): # return p(y=1|X), basically same as self.p
        # (n-by-m+1 * m+1-by-1) or (bs-by-m+1 * m+1-by-1)
        return(np.exp(np.dot(X,self.weights))/(1+np.exp(np.dot(X,self.weights))))
    '''
    
    ## start of Error section
    def computeError(self,Y,val=False):
        # use current p to compute log-loss (add regularization if needed). both training and validation
        # return a float value
        if val:
            logLoss = -np.sum([y*np.log(p) + (1-y)*np.log(1-p) for y,p in zip(Y,self.val_p)])
        else:
            logLoss = -np.sum([y*np.log(p) + (1-y)*np.log(1-p) for y,p in zip(Y,self.p)])
        logLoss = logLoss / len(Y)
        pen = self.mu*np.sum(self.weights[1:]**2) / len(Y) # should not include bias (according to Andrew Ng)
        return(logLoss+pen)
        
    def computeIndividualError(self,Y,val=False):
        # use current p to compute log-loss (add regularization if needed). both training and validation
        # return a numpy array of float value
        if val:
            logLoss = -([y*np.log(p) + (1-y)*np.log(1-p) for y,p in zip(Y,self.val_p)])
        else:
            logLoss = -([y*np.log(p) + (1-y)*np.log(1-p) for y,p in zip(Y,self.p)])
        pen = self.mu*np.sum(self.weights[1:]**2) 
        return(np.array(logLoss)+pen)
    
    def storeError(self,Y,val=False):
        '''
        a. Call compute error functions and store the result in e_array
        b. Also compare to previous validation error and save the best model
        '''
        # 0. compute error
        error = self.computeError(Y,val)
        # 1. append to array
        if val:
            # update best validation score
            if self.best_val is None or self.best_val > error:
                self.best_val = error
                # trigger to save the best model
                self.save = True
                
            # store validation error into e_array
            self.e_array[1].append(error) # validation
            
        else:
            self.e_array[0].append(error) # training

    def computeAverageError(self):
        '''
        Each entry is the average of past n_plot points
        '''
        # 0. check if we have passed n_plot points
        if self.dataSeen // self.n_plot > self.plotPoint:
            self.plotPoint = self.dataSeen//self.n_plot
            # 1. compute mean error array and remove it from the e-array
            self.error[0].append(np.mean(self.e_array[0][:self.n_plot]))
            self.e_array[0] = self.e_array[0][self.n_plot:]
            # 2. do the same thing for validation
            if self.validation:
                self.error[1].append(np.mean(self.e_array[1][:self.n_plot]))
                self.e_array[1] = self.e_array[1][self.n_plot:]
    
    def visulizeError(self,val=False): # plot training (and validation) graph
        '''
        Visualize the error(s) against the number of data seen.
        The scale of x-axis depends on n_plot
        '''
        plt.figure()
        plt.plot(self.error[0],label='Training error') # training
        if val:
            plt.plot(self.error[1],label='Validation error') # validation   
        plt.legend(loc='upper right')
        plt.title('Error plot')
        plt.ylabel('Error')
        plt.xlabel('Number of data seen (steps = '+str(self.n_plot)+')')
        plt.show()              
    
    ## end of Error section
    
    def computeDelta(self,X,Y): # compute gradient: 1-by-bs * bs-by-m+1
        self.gradients = - np.dot((Y-self.p).T,X).T 
        # (y-p)*x 

    def update(self): # use in fit, use compute gradient
        self.weights = self.weights - self.alpha * (self.gradients + 2*self.mu*self.weights) 
        # 2 can be taken out and incoporated into self.mu too
        # b_j := b_j + alpha* (gradient - regularization punishment)
        # => b_j + alpha*[(y_i-p_i) x_i - 2 mu*b_j]
            
    def classify(self,X,threshold=None):
        self.label = np.ones(X.shape[0]).reshape(-1,1)
        # prep X to be in the format we need
        if (type(X).__module__ != np.__name__):
            X = np.array(X)
        X = np.concatenate((np.ones((X.shape[0],1)),X),1)
        
        # predict using weights
        temp = np.exp(np.dot(X,self.weights))/(1+np.exp(np.dot(X,self.weights)))
        if threshold:
            self.label[temp<threshold] = 0
        else:
            self.label[temp<self.threshold] = 0
        return(self.label)
    
    def standardizeX(self,X):
        ### to standardize input X, change to numpy array and add intercept
        # 0. change X to np.array 
        if (type(X).__module__ != np.__name__):
            X = np.array(X)
        
        # 1. add intercept term
        X = np.concatenate((np.ones((X.shape[0],1)),X),1)
        return(X)
            
    def fit(self,X,Y,initialize=True,verbo=False,shuffle=True,val_split=0.0,val_data=None,filePath=None): 
        '''
        call with data, do the iteration, keep updating weights
        
        If provided with validation data (in tuple form), will use this as validation set.
        If no validation data but have val_split, then it will split X,Y into validation set
        When there is validation data, the time it takes to finish 'fit' function will depend 
        on how large the validation data is.
        
        # -1. standardized X input
        # 0. check if weights are initialized, if not, call the initialize function. reset parameters if needed
        # 0.5 validation data
        # 1. shuffle data
        # 1.5.1. if val_data provided, ignore split.
        # 1.5.2. if val_data not provided, split
        # 1.5.3. if no val_data no split, skip
        # 2. break X and y into batches, careful of last batch
        # 2.5 update the number of data the model have seen, for plotting error purposes 
        # 3. for each batch, predict -> gradient -> update
        # 3.1 before update, calculate the error
        '''
        
        # -1.
        X = self.standardizeX(X)
        
        # 0. initialize if needed
        if not hasattr(self,'weights') or initialize:
            self.initialize(X)
                    
        # 0.5. Validation set
        # validation graph must come with training error curve
        if val_data is not None:
            # use val_data
            self.validation, self.testing = True, True
            if len(val_data) == 2: # tuple of (X,Y)
                val_x, val_y = val_data
                val_x = self.standardizeX(val_x) # standardize val_x input
        elif 0.0< val_split < 1.0:
            # split X
            self.validation, self.testing = True, True
            split_at = np.floor(X.shape[0]*val_split).astype(int)
            val_x = X[:split_at,:]
            val_y = Y[:split_at]
            X = X[split_at:,:]
            Y = Y[split_at:]
            
        # 1. shuffle
        ind = np.arange(X.shape[0])
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(ind)
            
        # 2.
        for j in range(self.epoch):
            for i in range(0,X.shape[0],self.batch_size): # each batch                
                if verbo: print('batch starting at {}'.format(i));print('')
                
                if i+self.batch_size > X.shape[0]:
                    x = X[ind[i:X.shape[0]],:]
                    y = Y[ind[i:X.shape[0]]]
                else:
                    x = X[ind[i:i+self.batch_size],:]
                    y = Y[ind[i:i+self.batch_size]]
                
                # 2.5. update the number of data the model have seen   
                if self.testing:
                    self.dataSeen += x.shape[0]
                
                # 3.
                if verbo:
                    self.predict(x)
                    print("predict-self.p")
                    print(self.p.shape);print(self.p);print('')
                    print("delta-self.gradients")
                    if self.validation:
                        self.predict(val_x,val=True)
                        print("Validation error")
                        print(self.computeError(val_y,val=True));print('')
                        self.storeError(val_y,val=True)
                    print("Training Error")
                    print(self.computeError(y));print('')
                    if self.testing:
                        self.storeError(y)
                    self.computeDelta(x,y)
                    print(self.gradients.shape);print(self.gradients);print('')
                    print("update-self.weights")
                    self.update()
                    print(self.weights.shape);print(self.weights);print('')
                else:
                    # compute probability using existing weights -> use self.w to get self.p
                    self.predict(x) 
                    # compute validation error
                    if self.validation:
                        self.predict(val_x,val=True)
                        self.storeError(val_y,val=True)
                        if self.save: self.saveModel(filePath)
                    # compute training error
                    if self.testing:
                        self.storeError(y)
                        self.computeAverageError()
                    # compute gradient -> use self.p and y to get gradient
                    self.computeDelta(x,y)
                    # update weights -> use self.gradient to update self.w
                    self.update()   
                
                if np.isnan(self.weights).any() or np.isnan(self.p).any() or np.isnan(self.gradients).any():
                    break
    
    def _gradientCheck(self):
        # TO-DO
        pass
    
    def saveModel(self,filePath):
        '''
        should save the model and the weights, so it could be loaded directly and use without training
        '''
        if filePath is None:
            filePath = 'best-model.pkl'    
        if pickle is None:
            raise ImportError('save_model requires pickle')
        
        with open(filePath,'wb') as f:
            pickle.dump(self,f,-1) # highest protocol