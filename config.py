import numpy as np
from easydict import EasyDict as edict

model_training = edict()
model_training.dataset = 'cifar10' #options are ['cifar10', 'svhn']
model_training.model_id = 1

####### CIFAR-10 training configuration #########

cifar10_params = edict()
cifar10_params.num_classes = 10
cifar10_params.learning_rate = 0.05
cifar10_params.optimizer = 'sgd'
cifar10_params.schedule_epoch = [800, 1000] # [600,800] for mt, [800,1000] for lc
cifar10_params.device_ids = [0]
cifar10_params.batch_size = 128
cifar10_params.label_batch_size = 32
cifar10_params.num_workers = 4
cifar10_params.save_dir = 'experiment_results/models/'  #model_{}'.format(model_training.model_id)
cifar10_params.log_dir = 'experiment_results/logs/'
cifar10_params.valid_freq = 5
cifar10_params.train_data_label_percentage = 0.08
cifar10_params.label_random_seed = 2

# Use for mean-teacher method
cifar10_params.mt_consistency_weight = 100.0
cifar10_params.mt_ema_decay = 0.99
cifar10_params.mt_rampup_epochs = 5

# Use for mean_teacher_local_clustering
cifar10_params.lc_dist_threshold = 40.0 # 40.0 for 2000 labeled samples, and 50.0 for 4000 labeled samples
cifar10_params.lc_weight = 10.0
cifar10_params.lc_rampup_epochs = 100
cifar10_params.lc_start_loss_epoch = 600 # lc start from 600-th epoch


####### SVHN training configuration #########

svhn_params = edict()
svhn_params.num_classes = 10
svhn_params.learning_rate = 0.05
svhn_params.optimizer = 'sgd'
svhn_params.schedule_epoch = [400, 600] # [300,500] for mt, [400, 600] for lc
svhn_params.device_ids = [0]
svhn_params.batch_size = 128
svhn_params.label_batch_size = 32
svhn_params.num_workers = 4
svhn_params.save_dir = 'experiment_results/models/'
svhn_params.log_dir = 'experiment_results/logs/'
svhn_params.valid_freq = 5
svhn_params.train_num_label_examples = 1000 # options are 500, 1000
svhn_params.label_random_seed = 2

svhn_params.mt_consistency_weight = 100.0
svhn_params.mt_ema_decay = 0.995 # 0.995 for labeled samples 1000 and 500
svhn_params.mt_rampup_epochs = 5

svhn_params.lc_dist_threshold = 50.0
svhn_params.lc_weight = 20.0
svhn_params.lc_rampup_epochs = 50
svhn_params.lc_start_loss_epoch = 300 # lc start from 300-th epoch






