from data.opml import OPML
from tqdm import tqdm
import numpy as np
from layers.self_dis_torch import SelfDistill
from tqdm import tqdm
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK
import time
import logging
import openml
import torch.nn.functional as F

def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)

SEED = 42
np.random.seed(SEED)
SUITE = openml.study.get_suite(218)
ALL_TASKS = SUITE.tasks[:1]

# ALL_TASKS = [168868]


# [3, 12, 31, 53, 3917, 3945, 7592, 7593, 9952, 9977, 9981, 10101, 14965, 34539, 146195, 146212, 146606, 146818, 146821, 
# 146822, 146825, 167119, 167120, 168329, 168330, 168331, 168332, 168335, 168337, 168338, 168868, 168908, 168909, 168910, 
# 168911, 168912, 189354, 189355, 189356]
# ID =  168331 # Volkert
# assert ID in ALL_TASKS
N_STEP = 1
A=0.05
def objective(trainX, trainY, args):
    logger.info(args)
    net = SelfDistill(trainX,F.one_hot(trainY.long()),logger,alpha=A,**args)
    acc = net.distill(steps=N_STEP)
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


for ID in ALL_TASKS:
    rstate = np.random.default_rng(SEED)
    task = openml.tasks.get_task(ID) 
    FOLD = int(task.estimation_parameters['number_folds'])
    logger = get_logger(logger_name=f'{ID}',log_file=f'./log/{ID}.log')
    logger.info(f'{task}')

    TOTAL_ACC = []

    for f in tqdm(range(FOLD)):

        train_d, test_d = OPML(task,cv=f), OPML(task,cv=f,train=False)

        NC = train_d.N_TYPES

        search_space = {
            'lambd': hp.uniform('lambd',-12,6),
            'N': hp.quniform('N',128,2000,1),
            'scale': hp.uniform('scale',0,10),
            # 'alpha':hp.uniform('alpha',0,1),
        }


        best = fmin(
            fn = partial(objective,train_d.X, train_d.y),
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            rstate=rstate,
            verbose=True
        )

        best_net = SelfDistill(train_d.X,F.one_hot(train_d.y.long()),logger,alpha=A,**best)
        best_net.distill(steps=N_STEP)
        acc = best_net.predict(test_d.X,F.one_hot(test_d.y.long()))

        TOTAL_ACC.append(acc)
        logger.info(f'Accuracy for FOLD{f}\t{acc}')
        logger.info(f'Configuration for FOLD{f}\t{best}')

    logger.info('---'*5)
    logger.info(f'Accuracy\t{np.array(TOTAL_ACC).mean()}')
    logger.info(f'{TOTAL_ACC}')

# distiller = SelfDistill(train_d.X.numpy(),np.eye(NC)[train_d.y.numpy()])
# distiller.distill(alpha=0.5, steps=3)
# acc=distiller.predict(test_d.X.numpy(),np.eye(NC)[test_d.y.numpy()])

# print(acc)