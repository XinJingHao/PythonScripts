import numpy as np
import torch



class RunningMeanStd_tc:
    # calculate mean and std incrementally.
    # https://datagenetics.com/blog/november22017/index.html
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape)
        self.S = torch.zeros(shape) # S is a Intermediate variable, which is n*mean^2
        self.std = torch.zeros(shape)

    def update(self, x):
        x = torch.tensor(x)
        self.n += 1
        if self.n == 1: self.mean = x 
            # self.S and self.std of the first data are all 0, no need to set again.
            # By the way, the first data need not normalization, or you will encounter (x-mean)/std=0/0
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / (self.n))



class RunningMeanStd_np:
    # calculate mean and std incrementally.
    # https://datagenetics.com/blog/november22017/index.html
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape) # S is a Intermediate variable, which is n*mean^2
        self.std = np.zeros(shape)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1: self.mean = x 
            # self.S and self.std of the first data are all 0, no need to set again.
            # By the way, the first data need not normalization, or you will encounter (x-mean)/std=0/0
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / (self.n))
        


RN = RunningMeanStd_tc(1)  #pytorch version
RN = RunningMeanStd_np(1)  #numpy version

rlist = np.arange(1,6)

for r in rlist:
    RN.update(r)
    print('Mean:',RN.mean,'   Std:',RN.std, '\n')
    
print('Real mean: ',rlist.mean())
print('Real  std: ',rlist.std())
print('\n')











