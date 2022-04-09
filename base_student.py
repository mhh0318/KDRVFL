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



if __name__ == "__main__":
    # numpy.random.seed(0)
    torch.manual_seed(42)
    rstate = np.random.default_rng(42)

    ID = 31
    device='cpu'
    # D = UCIDataset('car')
    # trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = D.getitem(0)

    # task = openml.tasks.get_task(168331) 
    task = openml.tasks.get_task(ID) 
    print(task)
    res = []
    for f in range(10):
        # _, self.CV_NUM, _ = task.get_split_dimensions()
        train_d, test_d = OPML(task, cv=f), OPML(task, cv=f,train=False)
        # train_loader = DataLoader(dataset=train_d, batch_size=128,shuffle=True)
        # test_loader = DataLoader(dataset=test_d, batch_size=128,shuffle=True)

        NC = train_d.N_TYPES
        SEQ = train_d.SEQ


        search_space = {
            'C': hp.uniform('C',-12,6),
            'N': hp.quniform('N',128,2000,1),
            'tuning_vector': hp.uniform('tuning_vector',0,10),
        }


        def objective(trainX,trainY,args):
            net = RVFL_layer(classes=NC,args=args,device=device)
            trainY = F.one_hot(trainY).float()
            yhat = net.train(X=trainX.to(device),target=trainY.to(device))
            # yhat = net.eval(evalX.to(device))
            # loss = F.cross_entropy(yhat,trainY.argmax(1).to(device)).detach()
            loss = 1-((yhat.argmax(1).cpu() == trainY.argmax(1)).sum()/ len(yhat))*1.
            return {
                'loss': loss,
                'status': STATUS_OK,
                # -- store other results like this
                'eval_time': time.time(),
                'other_stuff': {'type': None, 'value': [0, 1, 2]},
                }

        # @torch.no_grad()
        # def distiller(model,test_loader):
        #     kd = []
        #     model.eval()
        #     for _,item in enumerate(tqdm(test_loader)):
        #         outputs_eval = model(item[0].cuda())
        #         kd.append(outputs_eval.cpu())
            
        #     kd = torch.cat(kd)
        #     return kd

        # model_path = '/home/hu/KDRVFL/id168331cv2.ckpt'
        # ckpt = torch.load(model_path)
        # teacher_model_args = dict(seq=SEQ,num_classes=NC,
        #     num_blocks=8, embed_dim=256, mlp_ratio=4,
        #     block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine)
        # teacher_model = MlpMixer(**teacher_model_args).cuda()
        # teacher_model.load_state_dict(ckpt)

        # kds = distiller(teacher_model, test_loader)

        best = fmin(
            fn = partial(objective,train_d.X, train_d.y),
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            rstate=rstate
        )
        best_net = RVFL_layer(classes=NC, args=best, device='cpu')
        _ = best_net.train(train_d.X.to(device),F.one_hot(train_d.y).float().to(device))
        yhat = best_net.eval(test_d.X.to(device))
        test_acc =  ((yhat.argmax(1).cpu().numpy() == test_d.y.numpy()).sum() / len(yhat))*100.
        print(test_acc)
        print(best)
        res.append(test_acc)
    print(res)
    print(np.array(res).mean())