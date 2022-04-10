# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.special import expit
from sklearn import metrics
from easydict import EasyDict
from data.uci import UCIDataset
import torch
import torch.nn as nn
from scipy import stats

class RVFL_layer(nn.Module):
    def __init__(self, classes, args, device, logger=None):
        super().__init__()
        self.args = EasyDict(args)
        self.args.lamb = 2**self.args.C
        self.classes = classes
        self.Params = EasyDict()
        self.d = device

        self.logger = logger
        # np.random.seed(32)
        # torch.manual_seed(32)
    # def sigmoid(self,x):
    #     return 1 / (1 + np.exp(-x))
    
        
    def ridge(self, x):
        n_sample, n_D = x.shape
        if n_D<n_sample:
            beta = torch.mm(torch.linalg.inv(torch.eye(x.shape[1]).to(self.d)/self.args.lamb+torch.mm(x.T,x)),x.T)
        else:
            beta = torch.mm(x.T,torch.linalg.inv(torch.eye(x.shape[0]).to(self.d)/self.args.lamb+torch.mm(x,x.T)))
        return beta

    def cal_pred(self, ridge, target, target_teacher):

        alpha = self.args.alpha

        beta = torch.mm(ridge,alpha*target+(1-alpha)*target_teacher)


        return beta

    
    def train(self, X, Y, steps):
        self._Y_distill = [Y]
        self._Y_pred = []
        self.beta_t = []
        alpha = self.args.alpha
        raw_X = X
        n_sample, n_D = X.shape
        # print(X.shape)
        w = (2 * torch.rand(int(self.args.N), n_D) - 1)  * self.args.tuning_vector
        b = torch.rand(1, int(self.args.N))
        w = w.T

        A_ = X @ w + b
        # layer normalization
        A_mean = torch.mean(A_, axis=0)
        A_std = torch.std(A_, axis=0)

        A_ = (A_ - A_mean) / A_std
        A_ = A_ + np.repeat(b, n_sample, 0)
        # A_ = args.gama * A_ + args.alpha

        A_ = torch.sigmoid(A_)


        A_merge = torch.cat([raw_X, A_, torch.ones((n_sample, 1))], axis=1)

        beta = []

        A_ridge = self.ridge(A_merge)
        # self.logger.info('---'*5)
        # self.logger.info('Step\tAccuracy')
        # self.logger.info('---'*5)
        for step in range(1, steps+1):
            beta_ = self.cal_pred(A_ridge, alpha*Y, (1-alpha)*self._Y_distill[step-1])
            # beta.append(beta_)
            self._Y_distill.append(torch.mm(A_merge, beta_))
            beta_pred =self.cal_pred(A_ridge, alpha*Y , (1-alpha)*self._Y_distill[step-1])
            Y_pred = torch.mm(A_merge, beta_pred)
            beta.append(beta_pred)
            self._Y_pred.append(Y_pred)
            # self.logger.info(f'{step}\t{self.acc(Y, Y_pred):1.5f}')
        hard_ensemble = stats.mode(torch.stack(self._Y_pred,0).argmax(2).numpy(),0)[0].squeeze()
        # self.logger.info(f'Ens\t{(Y.argmax(1).numpy()==hard_ensemble).mean():1.5f}')
        self.Params['w'] = w
        self.Params['b'] = b
        self.Params['beta'] = beta
        self.Params['mean'] = A_mean
        self.Params['std'] = A_std

        if A_merge.shape[1] < A_merge.shape[0]:
            self.beta_lim = torch.mm(torch.mm(torch.inverse(alpha*(A_merge.T@A_merge) + torch.eye(A_merge.shape[1])/self.args.lamb), alpha*A_merge.T), Y)
        else:
            self.beta_lim = torch.mm(torch.mm(alpha*A_merge.T, torch.inverse(alpha*(A_merge@A_merge.T) + torch.eye(A_merge.shape[0])/self.args.lamb)), Y)
        self.distilled_steps = steps
        # A_lim = torch.mm(A_merge,self.beta_lim)
        # self.logger.info(f'∞\t{self.acc(Y, A_lim):1.5f}')
        return (Y.argmax(1).numpy()==hard_ensemble).mean()
        # return (Y.argmax(1).numpy()==A_lim.argmax(1).numpy).mean()
    
    def eval(self, X, y, params=None):
        raw_X = X
        n_sample, n_D = raw_X.shape

        if params is not None:
            self.Params = params
        
        A_ = X @ self.Params.w + self.Params.b
        A_ = (A_ - self.Params.mean) / self.Params.std
        # A_ = (A_ - torch.mean(A_, axis=0)) / torch.std(A_, axis=0)
        A_ = A_ + np.repeat(self.Params.b, n_sample, 0)


        A_ = torch.sigmoid(A_)

        A_merge = torch.cat([raw_X, A_, torch.ones((n_sample, 1))], axis=1)


        self.logger.info('---'*5)
        self.logger.info('Test\tAccuracy')
        self.logger.info('---'*5)

        preds = []
        for i in range(self.distilled_steps):
            A_ = torch.mm(A_merge, self.Params.beta[i])
            preds.append(A_)
            self.logger.info(f'Step{i}\t{self.acc(y, A_):1.5f}')

        hard_ensemble = stats.mode(torch.stack(preds,0).argmax(2).numpy(),0)[0].squeeze()
        self.logger.info(f'HEns\t{(y.argmax(1).numpy()==hard_ensemble).mean():1.5f}')

        A_lim = torch.mm(A_merge,self.beta_lim)
        self.logger.info(f'∞\t{self.acc(y, A_lim):1.5f}')

        return (y.argmax(1).numpy()==hard_ensemble).mean() if (y.argmax(1).numpy()==hard_ensemble).mean()>(y.argmax(1).numpy()==A_lim.argmax(1).numpy()).mean() else (y.argmax(1).numpy()==A_lim.argmax(1).numpy()).mean()
        # return(y.argmax(1).numpy()==A_lim.argmax(1).numpy()).mean()

    def acc(self,x, Y_t):
        prediction = x.argmax(axis=1)
        return torch.mean((prediction == Y_t.argmax(1)).float())

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
