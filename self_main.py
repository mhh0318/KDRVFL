from data.opml import OPML
from tqdm import tqdm
import numpy as np
from layers.self_dis import SelfDistill
from tqdm import tqdm
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK
import time

SEED = 1
rstate = np.random.default_rng(SEED)
np.random.seed(SEED)

def objective(trainX,trainY,nc, args):
    net = SelfDistill(trainX,np.eye(nc)[trainY],**args)
    acc = net.distill(steps=10)
    loss = 1-acc
    # loss = F.mse_loss(yhat,trainY)
    # loss = F.mse_loss(yhat,trainY) + F.mse_loss(yhat, train_teacher)
    return {
        'loss': loss,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        }

# ID = 168329 
# ID = 146606 # Higgs
ID =  168331 # Volkert
train_d, test_d = OPML(id=ID,cv=1), OPML(id=ID,cv=1,train=False)

NC = train_d.N_TYPES

search_space = {
    'lambd': hp.uniform('lambd',-12,6),
    'N': hp.quniform('N',128,2000,1),
    'scale': hp.uniform('scale',0,10),
    'alpha':hp.uniform('alpha',0,1),
}


best = fmin(
    fn = partial(objective,train_d.X.numpy(), train_d.y.numpy(), NC),
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,
    rstate=rstate,
    verbose=True
)

best_net = SelfDistill(train_d.X.numpy(),np.eye(NC)[train_d.y],**best)
best_net.distill(steps=10)
acc = best_net.predict(test_d.X.numpy(),np.eye(NC)[test_d.y.numpy()])

print(acc)
print(best)
# distiller = SelfDistill(train_d.X.numpy(),np.eye(NC)[train_d.y.numpy()])
# distiller.distill(alpha=0.5, steps=3)
# acc=distiller.predict(test_d.X.numpy(),np.eye(NC)[test_d.y.numpy()])

# print(acc)