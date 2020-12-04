""" Main function for this repo. """
import argparse
import numpy as np
import torch
from misc import pprint
from meta import MetaTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters 
    parser.add_argument('--num_work', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', type=int, default=84)
    parser.add_argument('--model_type', type=str, default='conv4', choices=['conv4']) # The network architecture
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['miniImageNet', 'tieredImageNet', 'FC100']) # Dataset
    parser.add_argument('--phase', type=str, default='meta_train', choices=['meta_train', 'meta_eval']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--dataset_dir', type=str, default='/home/haoran/mini-imagenet/') # Dataset folder
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--index', type=int, default=0)

    ##### load ### 
    #parser.add_argument('--pre_load_path', type=str, default='/')
    

    # Parameters for meta-train phase
    parser.add_argument('--test_batch', type=int, default=6000)
    parser.add_argument('--val_batch', type=int, default=600) 
    parser.add_argument('--meta_batch', type=int, default=4) 
    parser.add_argument('--max_epoch', type=int, default=10000) # Epoch number for meta-train phase
    parser.add_argument('--num_batch', type=int, default=1000) # The number for different tasks used for meta-train
    parser.add_argument('--shot', type=int, default=1) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, default=5) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--meta_lr', type=float, default=0.01) 
    parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop
    parser.add_argument('--update_step', type=int, default=100) # The number of updates for the inner loop
    parser.add_argument('--eval_weights', type=str, default=None) # The meta-trained weights for meta-eval phase
    parser.add_argument('--meta_label', type=str, default='exp1') # Additional label for meta-train


    # Set and print the parameters
    args = parser.parse_args()
    pprint(vars(args))

    # Set manual seed for PyTorch
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Start trainer for pre-train, meta-train or meta-eval
    if args.phase=='meta_train':
        trainer = MetaTrainer(args)
        trainer.train()
    elif args.phase=='meta_eval':
        trainer = MetaTrainer(args)
        trainer.eval()
    else:
        raise ValueError('Please set correct phase.')
