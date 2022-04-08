from functools import partial
from layers.ResMLP import *
from data.opml import OPML
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from layers.ResMLP import *
from data.uci import UCIDataset
import openml


def om():

    # ID = 168331 #VOLKERT
    ID = 168329 #HELENA 
    # model = MlpMixer(**model_args)
    loss_fn = torch.nn.CrossEntropyLoss()

    accs = []
    for k in range(10):
        print('*****************************************CV*****************************************')

        train_d, test_d = OPML(id=ID, cv=k), OPML(id=ID, cv=k,train=False)
        train_loader = DataLoader(dataset=train_d, batch_size=128,shuffle=True)
        test_loader = DataLoader(dataset=test_d, batch_size=128,shuffle=True)
        seq = train_d.SEQ
        nc = train_d.N_TYPES
        model_args = dict(seq=seq,num_classes=nc,
            num_blocks=8, embed_dim=256, mlp_ratio=4,
            block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine)
        model = MlpMixer(**model_args).cuda()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters())
        best_loss = 10e6
        for i in tqdm(range(50)):
            # train
            running_loss = 0.
            running_acc = 0.
            for index,item in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(item[0].cuda())
                loss = loss_fn(outputs, item[1].cuda())
                loss.backward()
                optimizer.step()
                train_acc = (outputs.detach().argmax(1).cpu() == item[1]).sum()/len(item[1])*1.
                running_loss += loss.item()
                running_acc += train_acc
            epoch_loss = running_loss / (index+1)
            epoch_acc = running_acc / (index+1)
            # print(f"Training Epoch # {i} Loss is :{loss.item()}")
            # print(f"Training Acc # {i} is :{(outputs.detach().argmax(1) == torch.tensor(full_train_y).argmax(1)).sum()/len(full_train_x)*1.}")
            # test
            # if train_acc > best_acc:

            if epoch_loss < best_loss:
                running_eval_loss = 0.
                running_eval_acc = 0.
                with torch.no_grad():
                    model.eval()
                    for index,item in enumerate(test_loader):
                        outputs_eval = model(item[0].cuda())
                        loss_eval = loss_fn(outputs_eval,item[1].cuda())
                        acc = (outputs_eval.detach().argmax(1).cpu() == item[1]).sum()/len(item[1])*1.
                        running_eval_acc += acc
                        running_eval_loss += loss_eval

                epoch_eval_loss = running_eval_loss / (index+1)
                epoch_eval_acc = running_eval_acc / (index+1)

                print(f"Training Epoch # {i} Loss is :{epoch_loss}")
                print(f"Training Acc # {i} is :{epoch_acc}")
                print(f"Eval Epoch # {i} Loss is :{epoch_eval_loss}")
                print(f"Eval Acc # {i} is :{epoch_eval_acc}")
                best_loss = epoch_loss
                torch.save(model.state_dict(),f'id{ID}cv{k}.ckpt')
        accs.append(acc)
    print('ok')
    print(torch.tensor(accs))
    print(torch.tensor(accs).mean())




def uci():

    model_args = dict(seq=6,num_classes=4,
        num_blocks=2, embed_dim=64, mlp_ratio=2,
        block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine)
    # model = MlpMixer(**model_args)
    loss_fn = torch.nn.CrossEntropyLoss()
    D = UCIDataset('car')
    accs = []
    for k in range(4):
        print('*****************************************CV*****************************************')
        trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = D.getitem(k)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model = MlpMixer(**model_args).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        best_loss = 1e6
        for i in range(1000):
            # train
            optimizer.zero_grad()
            outputs = model(torch.tensor(full_train_x).float().cuda())
            loss = loss_fn(outputs, torch.tensor(full_train_y).argmax(1).long().cuda())
            loss.backward()
            optimizer.step()
            train_acc = (outputs.detach().argmax(1).cpu() == torch.tensor(full_train_y).argmax(1)).sum()/len(full_train_y)*1.
            # print(f"Training Epoch # {i} Loss is :{loss.item()}")
            # print(f"Training Acc # {i} is :{(outputs.detach().argmax(1) == torch.tensor(full_train_y).argmax(1)).sum()/len(full_train_x)*1.}")
            # test
            # if train_acc > best_acc:
            if loss.item() < best_loss:
                with torch.no_grad():
                    model.eval()
                    outputs_eval = model(torch.tensor(testX).float().cuda())
                    loss_eval = loss_fn(outputs_eval, torch.tensor(testY).argmax(1).long().cuda())
                    acc = (outputs_eval.detach().argmax(1).cpu() == torch.tensor(testY).argmax(1)).sum()/len(testY)*1.
                    # print(f"Eval Epoch # {i} Loss is :{loss_eval.item()}")
                    # print(f"Eval Acc # {i} is :{(outputs_eval.detach().argmax(1) == torch.tensor(testY).argmax(1)).sum()/len(testY)*1.}")

                    print(f"Training Epoch # {i} Loss is :{loss.item()}")
                    print(f"Training Acc # {i} is :{train_acc}")
                    print(f"Eval Epoch # {i} Loss is :{loss_eval.item()}")
                    print(f"Eval Acc # {i} is :{acc}")
                    # best_acc = train_acc
                    best_loss = loss.item()
                    # outputs_test = model(torch.tensor(testX).float())
                    # acc_test = (outputs_test.detach().argmax(1) == torch.tensor(testY).argmax(1)).sum()/len(testY)*1.
                    # print(f"********Test Acc # {i} is********** :{acc_test}")
        accs.append(acc)
    print('ok')

if __name__ == "__main__":
    om()