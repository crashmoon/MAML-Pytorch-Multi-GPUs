import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from meta import Meta
from    torch import optim
import plot
import json
import time
from    copy import deepcopy

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=600000)
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
argparser.add_argument('--imgc', type=int, help='imgc', default=3)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
argparser.add_argument('--weight_decay', type=float, default=1e-4)
args = argparser.parse_args()
print(args)

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

class Param:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '/home/haoran/meta/miniimagenet/'
    out_path = '/home/haoran/meta/one/adam_clip/'
    #root = '/home/haoran/meta/miniimagenet/'
    #root = '/storage/haoran/miniimagenet/'
    #root = '/disk/0/storage/haoran/miniimagenet/'
    root = '/mnt/hd/0/storage/haoran/miniimagenet/'   #change to your own root!#
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

if not os.path.exists(Param.out_path):
    os.makedirs(Param.out_path)

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def inf_get(train):
    while (True):
        for x in train:
            yield x


def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    test_result = {}
    best_acc = 0.0

    maml = Meta(args, Param.config).to(Param.device)
    maml = torch.nn.DataParallel(maml)
    opt = optim.Adam(maml.parameters(), lr=args.meta_lr)
    #opt = optim.SGD(maml.parameters(), lr=args.meta_lr, momentum=0.9, weight_decay=args.weight_decay)  

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    trainset = MiniImagenet(Param.root, mode='train', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, resize=args.imgsz)
    testset = MiniImagenet(Param.root, mode='test', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, resize=args.imgsz)
    trainloader = DataLoader(trainset, batch_size=args.task_num, shuffle=True, num_workers=4, drop_last=True)
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    train_data = inf_get(trainloader)
    test_data = inf_get(testloader)

    for epoch in range(args.epoch):
        support_x, support_y, meta_x, meta_y = train_data.__next__()
        support_x, support_y, meta_x, meta_y = support_x.to(Param.device), support_y.to(Param.device), meta_x.to(Param.device), meta_y.to(Param.device)
        meta_loss = maml(support_x, support_y, meta_x, meta_y).mean()
        opt.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_value_(maml.parameters(), clip_value = 10.0)
        opt.step()
        plot.plot('meta_loss', meta_loss.item())

        if(epoch % 1000 == 999):
            ans = None
            maml_clone = deepcopy(maml)
            for _ in range(25):
                support_x, support_y, qx, qy = test_data.__next__()
                support_x, support_y, qx, qy = support_x.to(Param.device), support_y.to(Param.device), qx.to(Param.device), qy.to(Param.device)
                temp = maml_clone(support_x, support_y, qx, qy, meta_train = False)
                if(ans is None):
                    ans = temp
                else:
                    ans = torch.cat([ans, temp], dim = 0)
            ans = ans.mean(dim = 0).tolist()
            test_result[epoch] = ans
            if (ans[-1] > best_acc):
                best_acc = ans[-1]
                torch.save(maml.state_dict(), Param.out_path + 'net_'+ str(epoch) + '_' + str(best_acc) + '.pkl') 
            del maml_clone
            print(str(epoch) + ': '+str(ans))
            with open(Param.out_path+'test.json','w') as f:
                json.dump(test_result,f)
        if (epoch < 5) or (epoch % 100 == 99):
            plot.flush()
        plot.tick()


os.chdir(Param.out_path)
if __name__ == '__main__':
    main()

