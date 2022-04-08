from functools import partial
from layers.ResMLP import *
from data.opml import OPML
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from layers.ResMLP import *
from layers.RVFL_dis import *
from data.uci import UCIDataset
import openml
from data.opml import OPML
from hyperopt import fmin, tpe, hp, STATUS_OK
from torch.nn import functional as F
import time
from tqdm import tqdm
import numpy



def main(seed=0):
    # numpy.random.seed(0)
    torch.manual_seed(seed)
    rstate = np.random.default_rng(seed)

    device='cpu'
    # D = UCIDataset('car')
    # trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = D.getitem(0)
    ID = 168329
    # task = openml.tasks.get_task(ID) 
    # _, self.CV_NUM, _ = task.get_split_dimensions()
    train_d, test_d = OPML(id=ID,cv=1), OPML(id=ID,cv=1,train=False)
    train_loader = DataLoader(dataset=train_d, batch_size=128,shuffle=False)
    # test_loader = DataLoader(dataset=test_d, batch_size=128,shuffle=True)

    NC = train_d.N_TYPES
    SEQ = train_d.SEQ


    search_space = {
        'C': hp.uniform('C',-12,6),
        'N': hp.quniform('N',128,2000,1),
        'tuning_vector': hp.uniform('tuning_vector',0,10),
        'alpha':hp.uniform('alpha',0,1),
    }

    def objective(trainX,trainY,train_teacher,args):
        net = RVFL_layer(classes=NC,args=args,device=device)
        trainY = F.one_hot(trainY).float()
        yhat = net.train(X=trainX.to(device),target=trainY.to(device),target_t=train_teacher)
        # yhat = net.eval(evalX.to(device))
        # loss = F.cross_entropy(yhat,trainY.argmax(1).to(device)).detach()
        loss = 1-((yhat.argmax(1).cpu() == trainY.argmax(1)).sum()/ len(yhat))*1.
        # loss = F.mse_loss(yhat,trainY)
        # loss = F.mse_loss(yhat,trainY) + F.mse_loss(yhat, train_teacher)
        # loss = args['alpha']*F.mse_loss(yhat,trainY) + (1-args['alpha'])*F.mse_loss(yhat, train_teacher)
        return {
            'loss': loss,
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'other_stuff': {'type': None, 'value': [0, 1, 2]},
            }

    @torch.no_grad()
    def distiller(model,loader):
        kd = []
        acc =0.
        loss = 0.
        criterion = nn.CrossEntropyLoss()
        model.eval()
        for index,item in enumerate(tqdm(loader)):
            outputs_eval = model(item[0].cuda())
            batch_loss = criterion(outputs_eval,item[1].cuda())
            kd.append(outputs_eval.cpu())
            batch_acc = (outputs_eval.argmax(1).cpu() == item[1]).sum() / len(item[1])
            acc += batch_acc
            loss += batch_loss
        acc /= index+1
        loss /= index+1
        print(f'eval_acc {acc}')
        print(f'eval_loss {loss}')
        kd = torch.cat(kd)
        return kd

    
    # model_path = f'/home/hu/KDRVFL/id{ID}cv2.ckpt'
    model_path = f'/home/hu/KDRVFL/id{ID}cv2_earlystop.ckpt'
    ckpt = torch.load(model_path)
    teacher_model_args = dict(seq=SEQ,num_classes=NC,
        num_blocks=8, embed_dim=256, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine)
    teacher_model = MlpMixer(**teacher_model_args).cuda()
    teacher_model.load_state_dict(ckpt)

    kds = distiller(teacher_model, train_loader)

    kds = F.softmax(kds,dim=-1)

    best = fmin(
        fn = partial(objective,train_d.X, train_d.y, kds),
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        rstate=rstate
    )
    best_net = RVFL_layer(classes=NC, args=best, device='cpu')
    _ = best_net.train(train_d.X.to(device),F.one_hot(train_d.y).float().to(device),kds)
    yhat = best_net.eval(test_d.X.to(device))
    test_acc =  ((yhat.argmax(1).cpu().numpy() == test_d.y.numpy()).sum() / len(yhat))*100.
    print(test_acc)
    print(best)
    return test_acc



def cv(seed=0,cv=0):
    # numpy.random.seed(0)
    torch.manual_seed(seed)
    rstate = np.random.default_rng(seed)

    device='cpu'
    # D = UCIDataset('car')
    # trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = D.getitem(0)
    ID = 168329
    # task = openml.tasks.get_task(ID) 
    # _, self.CV_NUM, _ = task.get_split_dimensions()
    train_d, test_d = OPML(id=ID,cv=cv), OPML(id=ID,cv=cv,train=False)
    train_loader = DataLoader(dataset=train_d, batch_size=128,shuffle=False)
    # test_loader = DataLoader(dataset=test_d, batch_size=128,shuffle=True)

    NC = train_d.N_TYPES
    SEQ = train_d.SEQ

    def objective(trainX,trainY,train_teacher,args):
        net = RVFL_layer(classes=NC,args=args,device=device)
        trainY = F.one_hot(trainY).float()
        yhat = net.train(X=trainX.to(device),target=trainY.to(device),target_t=train_teacher)
        # yhat = net.eval(evalX.to(device))
        # loss = F.cross_entropy(yhat,trainY.argmax(1).to(device)).detach()
        loss = 1-((yhat.argmax(1).cpu() == trainY.argmax(1)).sum()/ len(yhat))*1.
        # loss = F.mse_loss(yhat,trainY)
        # loss = F.mse_loss(yhat,trainY) + F.mse_loss(yhat, train_teacher)
        # loss = args['alpha']*F.mse_loss(yhat,trainY) + (1-args['alpha'])*F.mse_loss(yhat, train_teacher)
        return {
            'loss': loss,
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'other_stuff': {'type': None, 'value': [0, 1, 2]},
            }

    @torch.no_grad()
    def distiller(model,loader):
        kd = []
        acc =0.
        loss = 0.
        criterion = nn.CrossEntropyLoss()
        model.eval()
        for index,item in enumerate(tqdm(loader)):
            outputs_eval = model(item[0].cuda())
            batch_loss = criterion(outputs_eval,item[1].cuda())
            kd.append(outputs_eval.cpu())
            batch_acc = (outputs_eval.argmax(1).cpu() == item[1]).sum() / len(item[1])
            acc += batch_acc
            loss += batch_loss
        acc /= index+1
        loss /= index+1
        print(f'eval_acc {acc}')
        print(f'eval_loss {loss}')
        kd = torch.cat(kd)
        return kd



    search_space = {
        'C': hp.uniform('C',-12,6),
        'N': hp.quniform('N',128,2000,1),
        'tuning_vector': hp.uniform('tuning_vector',0,10),
        'alpha':hp.uniform('alpha',0,1),
    }


    # model_path = f'/home/hu/KDRVFL/id{ID}cv2.ckpt'
    model_path = f'/home/hu/KDRVFL/id{ID}cv2_earlystop.ckpt'
    ckpt = torch.load(model_path)
    teacher_model_args = dict(seq=SEQ,num_classes=NC,
        num_blocks=8, embed_dim=256, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine)
    teacher_model = MlpMixer(**teacher_model_args).cuda()
    teacher_model.load_state_dict(ckpt)

    kds = distiller(teacher_model, train_loader)

    # kds = F.softmax(kds,dim=-1)

    best = fmin(
        fn = partial(objective,train_d.X, train_d.y, kds),
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        rstate=rstate
    )
    best_net = RVFL_layer(classes=NC, args=best, device='cpu')
    _ = best_net.train(train_d.X.to(device),F.one_hot(train_d.y).float().to(device),kds)
    yhat = best_net.eval(test_d.X.to(device))
    test_acc =  ((yhat.argmax(1).cpu().numpy() == test_d.y.numpy()).sum() / len(yhat))*100.
    print(test_acc)
    print(best)
    return test_acc


if __name__ == '__main__':
    rs = [0,42,168,7316,86132]
    accs = []
    for i in range(5):
        test_accs = []
        for j in range(10):
            test_acc = cv(seed=rs[i],cv=j)
            test_accs.append(test_acc)
        print("*********************************************")
        print(test_accs)
        print(numpy.mean(test_accs))
        accs.append(test_accs)
    print("*********************************************")
    print(accs)
    print(numpy.mean(accs))

    # main(seed=0)