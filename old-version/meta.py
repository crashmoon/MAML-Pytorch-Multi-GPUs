import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
import  numpy as np
import  torch.autograd as autograd
from    copy import deepcopy

class Learner(nn.Module):
    def __init__(self, config, imgc, imgsz):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()
        self.config = config
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError
                
        for p in self.vars_bn:
            p.requires_grad = False
        pass

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        if vars is None:
            vars = self.vars
        idx = 0
        bn_idx = 0
        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError
        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x


    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


class Meta(nn.Module):
    def __init__(self, args, config):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.weight_decay = args.weight_decay
        self.net = Learner(config, args.imgc, args.imgsz)
        
    def single_task_loss(self, data_in):
        support_x, support_y, meta_x, meta_y = data_in
        meta_loss = []
        out = self.net(support_x)
        loss = F.cross_entropy(out, support_y)
        self.net.zero_grad()
        grad = autograd.grad(loss, self.net.parameters(), create_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
        meta_loss.append(F.cross_entropy(self.net(meta_x, vars=fast_weights), meta_y))
        for k in range(1, self.update_step):
            out = self.net(support_x, vars = fast_weights)
            loss = F.cross_entropy(out, support_y)
            self.net.zero_grad(vars = fast_weights)
            grad = autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            meta_loss.append(F.cross_entropy(self.net(meta_x, vars=fast_weights), meta_y))
        return meta_loss
            
    def forward(self, support_x, support_y, meta_x, meta_y, meta_train = True):
        if(meta_train):
            """
            :param support_x:   [b, setsz, c_, h, w]
            :param support_y:   [b, setsz]
            :param meta_x:      [b, setsz, c_, h, w]
            :param meta_y:      [b, setsz]
            """
            assert(len(support_x.shape) == 5)
            task_num_now = support_x.size(0)
            n_task_meta_loss = list(map(self.single_task_loss, zip(support_x, support_y, meta_x, meta_y)))
            re = n_task_meta_loss[0][-1].view(1,1)
            for i in range(1, task_num_now):
                re = torch.cat([re, n_task_meta_loss[i][-1].view(1,1)], dim = 0)
            return re  
        else:
            """
            :param support_x:   [b, setsz,   c_, h, w]
            :param support_y:   [b, setsz  ]
            :param qx:          [b, querysz, c_, h, w]
            :param qy:          [b, querysz]
            :return:            [b, acc_dim]
            """
            qx, qy = meta_x, meta_y
            assert(len(support_x.shape) == 5)
            task_num_now = qx.size(0)
            querysz = qx.size(1)
            ans = None
            for task_id in range(task_num_now):
                #net = deepcopy(self.net)
                fast_weights = self.net.parameters()
                acc = []
                for _ in range(self.update_step_test):
                    out = self.net(support_x[task_id], vars = fast_weights)
                    loss = F.cross_entropy(out, support_y[task_id])
                    self.net.zero_grad(vars = fast_weights)
                    grad = autograd.grad(loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))    

                    out_q = self.net(qx[task_id], vars = fast_weights)
                    __, predicted = out_q.max(1)
                    correct = predicted.eq(qy[task_id]).sum().item()
                    acc.append(correct/querysz)  
                #del net                  
                if (ans is None):
                    ans = torch.FloatTensor(acc).view(1,-1).to(support_x.device)
                else:
                    ans = torch.cat([ans, torch.FloatTensor(acc).view(1,-1).to(support_x.device)], dim = 0)
            return ans


    # def finetunning(self, support_x, support_y, qx, qy):
    #     """
    #     :param support_x:   [setsz,   c_, h, w]
    #     :param support_y:   [setsz  ]
    #     :param qx:          [querysz, c_, h, w]
    #     :param qy:          [querysz]
    #     :return:            acc
    #     """
    #     assert(len(support_x.shape) == 4)
    #     querysz = qx.size(0)
    #     net = deepcopy(self.net)
    #     fast_weights = net.parameters()
    #     acc = []
    #     for __ in range(self.update_step_test):
    #         out = net(support_x, vars = fast_weights)
    #         loss = F.cross_entropy(out, support_y)
    #         net.zero_grad(vars = fast_weights)
    #         grad = autograd.grad(loss, fast_weights)
    #         fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

    #         out_q = net(qx, vars = fast_weights)
    #         _, predicted = out_q.max(1)
    #         correct = predicted.eq(qy).sum().item()
    #         acc.append(correct/querysz)
    #     del net
    #     return torch.FloatTensor(acc)

