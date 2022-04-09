# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from math import sqrt
from scipy.special import expit
from sklearn import metrics
from easydict import EasyDict
from data.uci import UCIDataset
import torch
import torch.nn as nn

np.random.seed(32)
torch.manual_seed(32)



class RVFL_layer(nn.Module):
    def __init__(self, classes, args, device):
        super().__init__()
        self.args = EasyDict(args)
        self.args.lamb = 2**self.args.C
        self.classes = classes
        self.Params = EasyDict()
        self.d = device

        # np.random.seed(32)
        # torch.manual_seed(32)
    # def sigmoid(self,x):
    #     return 1 / (1 + np.exp(-x))
    
    def matrix_inverse_dis(self, x, target, target_teacher):
        n_sample, n_D = x.shape
        alpha = self.args.alpha
        if n_D<n_sample:
            beta = torch.mm(torch.mm(torch.linalg.inv(torch.eye(x.shape[1]).to(self.d)/self.args.lamb+torch.mm(x.T,x)),x.T),alpha*target+(1-alpha)*target_teacher)
        else:
            beta = torch.mm(x.T,torch.mm(torch.linalg.inv(torch.eye(x.shape[0]).to(self.d)/self.args.lamb+torch.mm(x,x.T)),alpha*target+(1-alpha)*target_teacher))

        return beta

    def matrix_inverse(self, x, target):
        n_sample, n_D = x.shape
        if n_D<n_sample:
            beta = torch.mm(torch.mm(torch.inverse(torch.eye(x.shape[1]).to(self.d)/self.args.lamb+torch.mm(x.T,x)),x.T),target)
        else:
            beta = torch.mm(x.T,torch.mm(torch.linalg.inv(torch.eye(x.shape[0]).to(self.d)/self.args.lamb+torch.mm(x,x.T)),target))

        return beta
    
    def train(self, X, target):
        raw_X = X
        n_sample, n_D = X.shape
        # print(X.shape)
        w = (2 * torch.rand(int(self.args.N), n_D) - 1)  * self.args.tuning_vector
        b = torch.rand(1, int(self.args.N))
        w = w.T
        self.w = w
            
        self.b = b

        A_ = X @ w + b
        # layer normalization
        A_mean = torch.mean(A_, axis=0)
        A_std = torch.std(A_, axis=0)
        self.mean = A_mean
        self.std = A_std

        A_ = (A_ - A_mean) / A_std
        A_ = A_ + np.repeat(b, n_sample, 0)
        # A_ = args.gama * A_ + args.alpha


        A_ = torch.sigmoid(A_)


        A_merge = torch.cat([raw_X, A_, torch.ones((n_sample, 1))], axis=1)

        self.A_ = A_
        beta_ = self.matrix_inverse(A_merge, target)
        self.beta = beta_
        # X = torch.cat([raw_X,  A_], axis=1)
        predict_score = A_merge @ beta_

        self.Params['w'] = self.w
        self.Params['b'] = self.b
        self.Params['beta'] = self.beta
        self.Params['mean'] = self.mean
        self.Params['std'] = self.std

        return predict_score
    
    def eval(self, X, params=None):
        raw_X = X
        n_sample, n_D = raw_X.shape

        if params is not None:
            self.Params = params
        
        A_ = X @ self.Params.w + self.Params.b
        # A_ = (A_ - self.Params.mean) / self.Params.std
        A_ = (A_ - torch.mean(A_, axis=0)) / torch.std(A_, axis=0)
        A_ = A_ + np.repeat(self.Params.b, n_sample, 0)


        A_ = torch.sigmoid(A_)
        
        
        self.A_t = A_

        A_merge = torch.cat([raw_X, A_, torch.ones((n_sample, 1))], axis=1)
        predict_score = A_merge @ self.Params.beta
        return predict_score
    
    def rvfl(self, previous_X, Y, previous_Xt, Yt):
        # train_score = softmax(self.train(previous_X, Y))
        # eval_score = softmax(self.eval(previous_Xt))
        train_score = self.train(previous_X, Y)
        eval_score = self.eval(previous_Xt)
        train_acc = np.mean(np.argmax(train_score,-1).ravel()==np.argmax(Y,axis=1))
        eval_acc = np.mean(np.argmax(eval_score,-1).ravel()==np.argmax(Yt,axis=1))
        # print(train_acc, eval_acc)
        return [train_acc, eval_acc], [train_score, eval_score]
    

if __name__ == '__main__':
    data = UCIDataset('abalone')
    trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = data.getitem(0)
    net_RVFL = RVFL_layer(classes=evalY.shape[1], args={'C':5, 'N':100, 'tuning_vector':2})
    net_RVFL.train(X=torch.tensor(trainX).float(),target=torch.tensor(trainY).float())
    yhat2 = net_RVFL.eval(torch.tensor(testX).float())
    acc_rvfl = ((yhat2.argmax(1).cpu().numpy() == testY.argmax(1)).sum() / len(testX))*100.
    print(acc_rvfl)
