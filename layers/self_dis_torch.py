import numpy as np
from scipy import stats
import torch
class Recurser(object):
    """Calculate B recursively based on provided A and alpha, according to
    recursive formula in paper."""
    def __init__(self, A, alpha):
        self.A = A
        self.B = np.copy(A)
        self.alpha = alpha
        
    def _update_step(self):
        self.B = np.matmul(self.A, (1-self.alpha)*self.B + self.alpha*np.identity(self.B.shape[0]))
    
    def update(self, steps=1):
        for i in range(steps):
            self._update_step()
        
    def get(self, val='B'):
        return getattr(self, val.upper())


class SelfDistill(object):
    def __init__(self, X, Y, logger=None, lambd=1e-6, N=256, scale=2, alpha=0.1, d='cpu'):
        self.X = X
        self.Y = Y
        self.lambd = 2**lambd
        self.alpha = alpha
        n_D = X.shape[1]

        N = int(N)

        self.w = (2 * torch.rand([N,n_D]) - 1) * scale
        self.b = torch.rand([1,N])
    
        self.logger = logger
        self.d = d
        self.distilled_steps = 0
        self._Y_distill = [Y]
        self._Y_pred = []
        self.beta_t = []
        
    def distill(self, steps=1, return_all=False):
        alpha = self.alpha
        assert np.isnan(self.X).sum() == 0
        self.K = self.random_mapping(self.X)

        if self.K.shape[1] < self.K .shape[0]:
            self.K_ridge = torch.mm(torch.inverse(torch.eye(self.K.shape[1]).to(self.d)/self.lambd+torch.mm(self.K.T,self.K)),self.K.T)
            # self.K_ridge = np.matmul(np.linalg.inv(np.matmul(self.K.T,self.K) + np.identity(self.K.shape[1])/self.lambd),self.K.T)
        else:
            self.K_ridge = torch.mm(self.K.T,torch.inverse(torch.eye(self.K.shape[0]).to(self.d)/self.lambd+torch.mm(self.K,self.K.T)))
            # self.K_ridge = np.matmul(self.K.T, np.linalg.inv(np.matmul(self.K,self.K.T) + np.identity(self.K.shape[0])/self.lambd))
        # Plot
        # plt.figure()
        # #plt.plot(x, np.sin(x*2*np.pi), 'k', linewidth=1, label='True func.')
        # plt.scatter(self.X, self.Y, marker='x', c='k', label=r'$\mathcal{D}_{\mathrm{train}}$')
        
        # Status print
        self.logger.info('---'*5)
        self.logger.info('Step\tAccuracy')
        self.logger.info('---'*5)
        
        # Perform distillation and plot/print
        for step in range(1, steps+1):
            # Calculate Y_t
            beta_t = torch.mm(self.K_ridge, alpha*self.Y + (1-alpha)*self._Y_distill[step-1])
            self.beta_t.append(beta_t)
            self._Y_distill.append(torch.mm(self.K, beta_t))
            
            # Calculate/predict with f_t on x

            # K_func_ridge = np.matmul(np.linalg.inv(self.K.T@self.K + self.lambd*np.identity(self.K.shape[1])),self.K.T)


            beta_pred =torch.mm(self.K_ridge, alpha*self.Y + (1-alpha)*self._Y_distill[step-1])
            Y_pred = torch.mm(self.K, beta_pred)
            self._Y_pred.append(Y_pred)

            # if steps < 11:
            #     plt.plot(x, Y_pred, label=f'$f_{{{step}}}$', color=my_cmap((step-1)/steps))
            # else:
            #     if step % (steps // 10) == 0:
            #         plt.plot(x, Y_pred, label=f'$f_{{{step}}}$', color=my_cmap((step-1) / (steps // 10)))
            
            # Print status
            self.logger.info(f'{step}\t{self.acc(self.Y, Y_pred):1.5f}')
        assert torch.isnan(torch.cat(self._Y_pred)).sum() == 0
        hard_ensemble = stats.mode(torch.stack(self._Y_pred,0).argmax(2).numpy(),0)[0].squeeze()
        self.logger.info(f'Ens\t{(self.Y.argmax(1).numpy()==hard_ensemble).mean():1.5f}')


        # # Calculate limiting solution
        # if self.K.shape[1] < self.K.shape[0]:
        #     beta_lim = np.matmul(np.matmul(np.linalg.inv(alpha*(self.K.T@self.K) + np.identity(self.K.shape[1])/self.lambd), alpha*self.K.T), self.Y)
        # else:
        #     beta_lim = np.matmul(np.matmul(alpha*self.K.T, np.linalg.inv(alpha*(self.K@self.K.T) + np.identity(self.K.shape[0])/self.lambd)), self.Y)

        # Y_lim = np.matmul(self.K, beta_lim)
        # # plt.plot(x, Y_lim, 'k--', label='$f_{\infty}$')
        # self.Y_lim = Y_lim

        # self.beta_lim = beta_lim
        
        # # Finalize plot
        # # plt.ylim(-1.5, 1.5)
        # # plt.legend(loc='lower left', ncol=steps+2 if steps < 11 else 10+2, mode='expand')
        
        # # print('---'*5)
        # self.logger.info(f'∞\t{self.acc(self.Y, Y_lim):1.5f}')


        self.distilled_steps = steps
        if return_all:
            return self._Y_pred
        # return self.acc(self.Y, Y_lim) if self.acc(self.Y,Y_lim) > (self.Y.argmax(1)==hard_ensemble).mean() else (self.Y.argmax(1)==hard_ensemble).mean()
        return (self.Y.argmax(1).numpy()==hard_ensemble).mean()
        # return self.acc(self.Y, Y_lim)
    
    def predict(self, X,y):
        self.logger.info('---'*5)
        self.logger.info('Test\tAccuracy')
        self.logger.info('---'*5)
        A = self.random_mapping(X)

        # Self KD steps:
        preds = []
        for i in range(self.distilled_steps):
            A_ = torch.mm(A, self.beta_t[i])
            preds.append(A_)
            self.logger.info(f'Step{i}\t{self.acc(y, A_):1.5f}')
            
        hard_ensemble = stats.mode(torch.stack(preds,0).argmax(2).numpy(),0)[0].squeeze()
        self.logger.info(f'HEns\t{(y.argmax(1).numpy()==hard_ensemble).mean():1.5f}')

        # A = np.matmul(A, self.beta_lim)
        # self.logger.info(f'∞\t{self.acc(y, A):1.5f}')

        # return self.acc(A,y) if self.acc(A,y) > (y.argmax(1)==hard_ensemble).mean() else (y.argmax(1)==hard_ensemble).mean()
        return(y.argmax(1).numpy()==hard_ensemble).mean()
        
    def random_mapping(self, X):

        A = X@self.w.T + self.b

        A_mean = torch.mean(A, axis=0)
        A_std = torch.std(A, axis=0)
        A = (A - A_mean) / A_std

        # A = A + np.repeat(self.b, X.shape[0], 0)

        A = torch.sigmoid(A)

        A_merge = torch.cat([X, A, torch.ones((X.shape[0], 1))], axis=1)

        return A_merge

    def acc(self,x, Y_t):
        prediction = x.argmax(axis=1)
        return torch.mean((prediction == Y_t.argmax(1)).float())

    
    def get_B(self, return_all=False):
        self.V, self.D, self.VT = np.linalg.svd(self.K, hermitian=True)
        self.D = np.diag(self.D)
        self.A = np.matmul(self.D, np.linalg.inv(self.D + self.lambd*np.identity(self.D.shape[0])))
        self.B_recurser = Recurser(self.A, self.alpha)
        
        # Calculate recursive B
        if return_all:
            Bs = []
            Bs.append(np.copy(self.B_recurser.get('B')))
            for i in range(1,self.distilled_steps):
                self.B_recurser.update(steps=1)
                Bs.append(np.copy(self.B_recurser.get('B')))
            
            # Get limiting B:
            self.B_lim = np.matmul(self.alpha*self.D, np.linalg.inv(self.alpha*self.D + self.lambd*np.identity(self.D.shape[0])))
            return(Bs, self.B_lim)
        else:
            self.B_recurser.update(steps=self.distilled_steps)
            return(np.copy(self.B_recurser.get('B')))