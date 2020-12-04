""" Generate commands for meta-train phase. """
import os
import math

def run_exp(num_batch=100, shot=1, query=15, lr=0.01, base_lr=0.01, update_step=5, meta_label = 'exp1', index = 0):
    max_epoch = 660
    way = 5
       
    the_command = (
        'python3 main.py' 
        + ' --max_epoch=' + str(max_epoch) 
        + ' --num_batch=' + str(num_batch) 
        + ' --shot=' + str(shot) 
        + ' --train_query=' + str(query) 
        + ' --way=' + str(way) 
        + ' --meta_lr=' + str(lr) 
        + ' --base_lr=' + str(base_lr) 
        + ' --update_step=' + str(update_step)                         
        + ' --dataset_dir=' + '/home/haoran/mini-imagenet/'            
        # + ' --dateset_dir=' + '/path/to/your/dataset/'        ## need edit ##
        + ' --meta_label=' + meta_label
        + ' --index=' + str(index)
        )

    os.system(the_command + ' --phase=meta_train')
    os.system(the_command + ' --phase=meta_eval')


####
run_exp(num_batch=1000, shot=1, query=15, lr=0.001, base_lr=0.01, update_step=5, meta_label = 'exp', index=0)

#######
# baseline 
# for idx in range(10):
#     run_exp(num_batch=1000, shot=1, query=15, lr=0.001, base_lr=0.01, update_step=5, meta_label = 'conv4-baseline', index=idx)


