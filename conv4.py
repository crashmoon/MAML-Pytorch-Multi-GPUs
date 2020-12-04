import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
import math


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        #self.weight.fast = None #Lazy hack to add fast weight link
        #self.bias.fast = None

    def forward(self, x):
        #if self.weight.fast is not None and self.bias.fast is not None:
        if hasattr(self.weight, 'fast') and hasattr(self.bias, 'fast'):
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = False):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        #self.weight.fast = None
        #if not self.bias is None:
        #    self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            #if self.weight.fast is not None:
            if hasattr(self.weight, 'fast'):
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            #if self.weight.fast is not None and self.bias.fast is not None:
            if hasattr(self.weight, 'fast') and hasattr(self.bias, 'fast'):
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            

#@weak_module
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        #self.weight.fast = None
        #self.bias.fast = None

    #@weak_script_method
    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum                
            pass
        pass

        #if self.weight.fast is not None and self.bias.fast is not None:
        if hasattr(self.weight, 'fast') and hasattr(self.bias, 'fast'):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight.fast, self.bias.fast,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)    

##########################################################################
################## bn fix running_mean ###################################
##########################################################################
#
# class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
#     def __init__(self, num_features):
#         super(BatchNorm2d_fw, self).__init__(num_features)
#         #self.weight.fast = None
#         #self.bias.fast = None

#     def forward(self, x):
#         running_mean = torch.zeros(x.data.size()[1]).cuda()
#         running_var = torch.ones(x.data.size()[1]).cuda()
#         #if self.weight.fast is not None and self.bias.fast is not None:
#         if hasattr(self.weight, 'fast') and hasattr(self.bias, 'fast'):
#             out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
#             #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
#         else:
#             out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
#         return out
#
##########################################################################

class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
        self.BN     = BatchNorm2d_fw(outdim)        
        self.relu   = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)
        for layer in self.parametrized_layers:
            self.init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def init_layer(self, L):
        # Initialization using fan-in
        if isinstance(L, nn.Conv2d) or isinstance(L, Conv2d_fw):
            n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
            L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
        elif isinstance(L, nn.BatchNorm2d) or isinstance(L, BatchNorm2d_fw):
            L.weight.data.fill_(1)
            L.bias.data.fill_(0)

    def forward(self,x):
        out = self.trunk(x)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)


class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out

def Conv4():
    return ConvNet(4)