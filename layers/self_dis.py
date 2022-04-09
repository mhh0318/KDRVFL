import numpy as np
from scipy import stats
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
    def __init__(self, X, Y, lambd=1e-6, N=256, scale=2, alpha=0.1):
        self.X = X
        self.Y = Y
        self.lambd = 2**lambd
        self.alpha = alpha
        n_D = X.shape[1]

        N = int(N)

        self.w = (2 * np.random.random([N,n_D]) - 1) * scale
        self.b = np.random.random([1,N])
    
        
        self.distilled_steps = 0
        self._Y_distill = [Y]
        self._Y_pred = []
        self.beta_t = []
        
    def distill(self, steps=1, return_all=False):
        alpha = self.alpha

        self.K = self.random_mapping(self.X)

        if self.K.shape[1] < self.K .shape[0]:
            self.K_ridge = np.matmul(np.linalg.inv(self.K.T@self.K + self.lambd*np.identity(self.K.shape[1])),self.K.T)
        else:
            self.K_ridge = np.matmul(self.K.T, np.linalg.inv(self.K@self.K.T + self.lambd*np.identity(self.K.shape[0])))
        # Plot
        # plt.figure()
        # #plt.plot(x, np.sin(x*2*np.pi), 'k', linewidth=1, label='True func.')
        # plt.scatter(self.X, self.Y, marker='x', c='k', label=r'$\mathcal{D}_{\mathrm{train}}$')
        
        # Status print
        print('---'*5)
        print('Step\tAccuracy')
        print('---'*5)
        
        # Perform distillation and plot/print
        for step in range(1, steps+1):
            # Calculate Y_t
            beta_t = np.matmul(self.K_ridge, alpha*self.Y + (1-alpha)*self._Y_distill[step-1])
            self.beta_t.append(beta_t)
            self._Y_distill.append(np.matmul(self.K, beta_t))
            
            # Calculate/predict with f_t on x

            # K_func_ridge = np.matmul(np.linalg.inv(self.K.T@self.K + self.lambd*np.identity(self.K.shape[1])),self.K.T)


            beta_pred = np.matmul(self.K_ridge, alpha*self.Y + (1-alpha)*self._Y_distill[step-1])
            Y_pred = np.matmul(self.K, beta_pred)
            self._Y_pred.append(Y_pred)

            # if steps < 11:
            #     plt.plot(x, Y_pred, label=f'$f_{{{step}}}$', color=my_cmap((step-1)/steps))
            # else:
            #     if step % (steps // 10) == 0:
            #         plt.plot(x, Y_pred, label=f'$f_{{{step}}}$', color=my_cmap((step-1) / (steps // 10)))
            
            # Print status
            print(f'{step}\t{self.acc(self.Y, Y_pred):1.5f}')
        
        hard_ensemble = stats.mode(np.array(self._Y_pred).argmax(2),0)[0].squeeze()
        print(f'Ensemble\t{(self.Y.argmax(1)==hard_ensemble).mean():1.5f}')
        # Calculate limiting solution
        if self.K.shape[1] < self.K.shape[0]:
            beta_lim = np.matmul(np.matmul(np.linalg.inv(alpha*(self.K.T@self.K) + self.lambd*np.identity(self.K.shape[1])), alpha*self.K.T), self.Y)
        else:
            beta_lim = np.matmul(np.matmul(alpha*self.K.T, np.linalg.inv(alpha*(self.K@self.K.T) + self.lambd*np.identity(self.K.shape[0]))), self.Y)

        Y_lim = np.matmul(self.K, beta_lim)
        # plt.plot(x, Y_lim, 'k--', label='$f_{\infty}$')
        self.Y_lim = Y_lim

        self.beta_lim = beta_lim
        
        # Finalize plot
        # plt.ylim(-1.5, 1.5)
        # plt.legend(loc='lower left', ncol=steps+2 if steps < 11 else 10+2, mode='expand')
        
        # print('---'*5)
        print(f'∞\t{self.acc(self.Y, Y_lim):1.5f}')
        self.distilled_steps = steps
        if return_all:
            return self._Y_pred
        return self.acc(self.Y, Y_lim)
    
    def predict(self, X,y):
        print('---'*5)
        print('Test\tAccuracy')
        print('---'*5)
        A = self.random_mapping(X)

        # Self KD steps:
        preds = []
        for i in range(self.distilled_steps):
            A_ = np.matmul(A, self.beta_t[i])
            preds.append(A_)
            print(f'Step{i}\t{self.acc(y, A_):1.5f}')
        hard_ensemble = stats.mode(np.array(preds).argmax(2),0)[0].squeeze()
        print(f'HEns\t{(y.argmax(1)==hard_ensemble).mean():1.5f}')
        A = np.matmul(A, self.beta_lim)
        print(f'∞\t{self.acc(y, A):1.5f}')
        return self.acc(A,y) if self.acc(A,y) < (y.argmax(1)==hard_ensemble).mean() else (y==hard_ensemble).mean()
        
    def random_mapping(self, X):

        A = X@self.w.T + self.b
        A = self.sigmoid(A)

        A_mean = np.mean(A, axis=0)
        A_std = np.std(A, axis=0)
        A = (A - A_mean) / A_std

        A = A + np.repeat(self.b, X.shape[0], 0)
        return A

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def acc(self,x, Y_t):
        prediction = x.argmax(axis=1)
        return np.mean(prediction == Y_t.argmax(1))

    def loss(self, x, Y_t):
        return np.mean(np.square(Y_t - np.sin(x*2*np.pi)))
    
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