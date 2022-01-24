import os
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
from utils import source_import

# ================
# LOAD CONFIGURATIONS

data_root = {'ImageNet': '/gpfs/scratch/lnsmith/deepLearning/data/imagenet',
             'Places': '/gpfs/scratch/lnsmith/deepLearning/data/places365'}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/Imagenet_LT/Stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--cyclical_factor', type=float, default=2,
                    help='1->Modified focal loss, 2->Cyclical focal loss (default=2)')
parser.add_argument('--gamma0', type=int, default=2,
                    help='Cyclical focal loss gamma (default=0)')
parser.add_argument('--gamma_pos', type=int, default=0,
                    help='Asymetric focal loss positive gamma (default=0)')
parser.add_argument('--gamma_neg', type=int, default=4,
                    help='Asymetric focal loss negative gamma (default=4)')
args = parser.parse_args()

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits

config = source_import(args.config).config
training_opt = config['training_opt']
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])
if args.config.find('cfl') > -1:
    config['criterions']['PerformanceLoss']['loss_params']['gamma_pos'] = args.gamma_pos
    config['criterions']['PerformanceLoss']['loss_params']['gamma_neg'] = args.gamma_neg
    config['criterions']['PerformanceLoss']['loss_params']['gamma0'] = args.gamma0
    config['criterions']['PerformanceLoss']['loss_params']['factor'] = args.cyclical_factor
    indx1 = args.config.find("config/") + 7
    indx2 = args.config[indx1:].find("/") 
    dataset = args.config[indx1:indx1+indx2] 
    if training_opt['log_dir'].find("meta") > -1:
        training_opt['log_dir'] = './logs/'+dataset+'/meta_embedding'+'_cfl_G'+str(args.gamma0)+str(args.gamma_pos)+str(args.gamma_neg)
    else:
        training_opt['log_dir'] = './logs/'+dataset+'/stage1'+'_cfl_G'+str(args.gamma0)+str(args.gamma_pos)+str(args.gamma_neg)
    print(" training_opt['log_dir']= ",  training_opt['log_dir'])
    if not os.path.exists(training_opt['log_dir']):
        os.mkdir(training_opt['log_dir'])
pprint.pprint(config)

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(), 
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x, 
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'])
            for x in (['train', 'val', 'train_plain'] if relatin_opt['init_centroids'] else ['train', 'val'])}

    training_model = model(config, data, test=False)

    training_model.train()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False)
            for x in ['train', 'test']}

    
    training_model = model(config, data, test=True)
    training_model.load_model()
    training_model.eval(phase='test', openset=test_open)
    
    if output_logits:
        training_model.output_logits(openset=test_open)
        
print('ALL COMPLETED.')
