from functools import partial
from layers.ResMLP import *
from data.opml import OPML
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from layers.ResMLP import *
from layers.RVFL import *
from data.uci import UCIDataset
import openml
from data.opml import OPML
from hyperopt import fmin, tpe, hp, STATUS_OK
from torch.nn import functional as F
import time
from tqdm import tqdm
import numpy
import torch

import logging


def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)

if __name__ == "__main__":
    # numpy.random.seed(0)
    SEED = 42
    np.random.seed(SEED)
    SUITE = openml.study.get_suite(218)
    ALL_TASKS = SUITE.tasks

    device='cpu'
    # D = UCIDataset('car')
    # trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = D.getitem(0)

    # task = openml.tasks.get_task(168331) 
    for ID in ALL_TASKS:
        rstate = np.random.default_rng(SEED)
        task = openml.tasks.get_task(ID) 
        FOLD = int(task.estimation_parameters['number_folds'])
        logger = get_logger(logger_name=f'{ID}',log_file=f'./log_base/{ID}.log')
        logger.info(f'{task}')
        print(task)
        TOTAL_ACC = [] 
        for f in range(FOLD):

            train_d, test_d = OPML(task,cv=f), OPML(task,cv=f,train=False)

            NC = train_d.N_TYPES
            SEQ = train_d.SEQ

            search_space = {
                'C': hp.uniform('C',-12,6),
                'N': hp.quniform('N',128,2000,1),
                'tuning_vector': hp.uniform('tuning_vector',0,10),
            }

            def objective(trainX,trainY,args):
                net = RVFL_layer(classes=NC,args=args,device=device)
                trainY = F.one_hot(trainY.long()).float()
                yhat = net.train(X=trainX.to(device),target=trainY.to(device))
                loss = 1-((yhat.argmax(1).cpu() == trainY.argmax(1)).sum()/ len(yhat))*1.
                return {
                    'loss': loss,
                    'status': STATUS_OK,
                    # -- store other results like this
                    'eval_time': time.time(),
                    'other_stuff': {'type': None, 'value': [0, 1, 2]},
                    }

            best = fmin(
                fn = partial(objective,train_d.X, train_d.y),
                space=search_space,
                algo=tpe.suggest,
                max_evals=20,
                rstate=rstate
            )
            best_net = RVFL_layer(classes=NC, args=best, device='cpu')
            _ = best_net.train(train_d.X.to(device),F.one_hot(train_d.y.long()).float().to(device))
            yhat = best_net.eval(test_d.X.to(device))
            test_acc =  ((yhat.argmax(1).cpu().numpy() == test_d.y.numpy()).sum() / len(yhat))
            TOTAL_ACC.append(test_acc)
            logger.info(f'Accuracy for FOLD{f}\t{test_acc}')
            logger.info(f'Configuration for FOLD{f}\t{best}')
        logger.info('---'*5)
        logger.info(f'Accuracy\t{np.array(TOTAL_ACC).mean()}')