import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data
from src.core import AppConfig
from src.data.leaf_loader import load_leaf_datasets, load_leaf_splits_datasets

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedsgd', 'fedavg', 'finetuning',
              'l2sgd', 'ditto', 'ewc', 'apfl', 'mapper', 'kl', 'meta']
DATASETS = ['vehicle', 'femnist', 'fmnist', 'celeba']   # fmnist: fashion mnist 


MODEL_PARAMS = {
    'fmnist.cnn': (10,), # num_classes
    'femnist.cnn': (62, ), # num_classes
    'vehicle.svm': (2, ), # num_classes
    'celeba.cnn': (2,), # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer',
                        type=str,
                        choices=OPTIMIZERS,
                        default='qffedavg')
    parser.add_argument('--dataset',
                        help='name of dataset',
                        type=str,
                        choices=DATASETS,
                        default='nist')
    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        default='stacked_lstm.py')
    parser.add_argument('--num_rounds',
                        help='number of communication rounds to simulate',
                        type=int,
                        default=-1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ communication rounds',
                        type=int,
                        default=-1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per communication round',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size of local training',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for the inner solver',
                        type=float,
                        default=0.003)
    parser.add_argument('--seed',
                        help='seed for random initialization',
                        type=int,
                        default=0)
    parser.add_argument('--sampling',
                        help='client sampling methods',
                        type=int,
                        default='2')  
    parser.add_argument('--q',
                        help='reweighting factor',
                        type=float,
                        default='0.0') 
    parser.add_argument('--data_partition_seed',
                        help='seed for splitting data into train/test/validation; default is train/test',
                        type=int,
                        default=0)
    parser.add_argument('--comm_freq',
                        help='the probabilities of communicating with server each round',
                        type=float,
                        default=0.1)
    parser.add_argument('--lambda_l2sgd',
                        help='how close those local models are',
                        type=float,
                        default=0)
    parser.add_argument('--num_corrupted',
                        help='how many corrupted devices',
                        type=int,
                        default=0)
    parser.add_argument('--boosting',
                        help='whether do model placement',
                        type=int,
                        default=0)
    parser.add_argument('--gradient_clipping',
                        help='whether do gradient clipping',
                        type=int,
                        default=0)
    parser.add_argument('--median',
                        help='whether do coordinate-wise median aggregation',
                        type=int,
                        default=0)
    parser.add_argument('--krum',
                        help='whether use krum to aggregate',
                        type=int,
                        default=0)
    parser.add_argument('--mkrum',
                        help='whether use multi-krum to aggregate',
                        type=int,
                        default=0)
    parser.add_argument('--k_norm',
                        help='whether to use k-norm aggregator',
                        type=int,
                        default=0)
    parser.add_argument('--k_loss',
                        help='whether to use k-loss aggregator',
                        type=int,
                        default=0)
    parser.add_argument('--fedmgda',
                        help='whether to use the fedmgda aggregator',
                        type=int,
                        default=0)
    parser.add_argument('--fedmgda_eps',
                        help='the epsilon parameter for fedmgda+',
                        type=float,
                        default=1.0)
    parser.add_argument('--random_updates',
                        help='whether send random updates',
                        type=int,
                        default=0)
    parser.add_argument('--local_iters',
                        help='number of local iterations',
                        type=int,
                        default=10)
    parser.add_argument('--alpha',
                        help='the alpha paramter in APFL',
                        type=float,
                        default=0.5)
    parser.add_argument('--lam',
                        help='lambda in the objective',
                        type=float,
                        default=0)
    parser.add_argument('--dynamic_lam',
                        help='whether device-specific lam',
                        type=int,
                        default=0)
    parser.add_argument('--global_reg',
                        help='global reg parameter for global model updates (not used for now)',
                        type=float,
                        default=-1)
    parser.add_argument('--finetune_iters',
                        help='finetune for how many epochs of sgd',
                        type=int,
                        default=100)
    parser.add_argument('--decay_factor',
                        help='learning rate decay for finetuning',
                        type=float,
                        default=1.0) # don't decay

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    if parsed['optimizer'] in ['l2sgd', 'ditto', 'apfl', 'mapper', 'ewc', 'meta', 'kl']:
        opt_path = 'flearn.trainers_MTL.%s' % parsed['optimizer']
    else:
        opt_path = 'flearn.trainers_global.%s' % parsed['optimizer']

    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]


    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main(cfg: AppConfig = None):
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()
    
    # Override options with cfg if provided
    if cfg:
        options['dataset'] = cfg.dataset.name
        options['batch_size'] = cfg.client.batch_size
        options['num_rounds'] = cfg.num_rounds
        options['clients_per_round'] = cfg.dataset.clients_per_round if cfg.dataset.clients_per_round else 10
        options['learning_rate'] = cfg.server.inner_lr
        options['num_epochs'] = cfg.client.local_epochs
        options['seed'] = cfg.seed
        
        # Load LEAF data
        print(f"Loading data from {cfg.dataset.root}...")
        # Prefer explicit train/val routing if available; fall back to legacy json_files
        train_files = getattr(cfg.dataset, 'train_json_files', None) or getattr(cfg.dataset, 'json_files', None)
        train_datasets = load_leaf_datasets(
            root=cfg.dataset.root,
            split="train",
            limit=cfg.dataset.num_clients,
            files=train_files
        )
        
        val_files = getattr(cfg.dataset, 'val_json_files', None)
        test_files = val_files or ([f.replace("train", "test", 1) for f in train_files] if train_files else None)
            
        try:
            test_datasets = load_leaf_datasets(
                root=cfg.dataset.root,
                split="test",
                limit=cfg.dataset.num_clients,
                files=test_files
            )
        except Exception as e:
            print(f"Warning: Failed to load test data: {e}")
            test_datasets = {}
            
        # Ditto expects dataset as tuple (users, groups, train_data, test_data)
        # train_data and test_data are dicts: {user: {'x': ..., 'y': ...}}
        # But load_leaf_datasets returns TensorDatasets.
        # We need to convert them or adapt Ditto's trainer.
        # Ditto uses TensorFlow!
        # Wait, Ditto implementation here uses TensorFlow?
        # "import tensorflow as tf" is present.
        # But HyperQLoRA is PyTorch based.
        # If Ditto is TF based, we cannot easily integrate it with PyTorch data loaders and models.
        # Let's check if Ditto has a PyTorch implementation or if we need to convert data.
        
        # If Ditto is TF, we might have a problem sharing the same environment/GPU resources efficiently if not careful.
        # But more importantly, the data format.
        
        # Let's check flearn/models/femnist/cnn.py to see if it is TF or PyTorch.
        pass

        # Load LEAF data via unified routing
        print(f"Loading data from {cfg.dataset.root}...")
        train_files = getattr(cfg.dataset, 'train_json_files', None) or getattr(cfg.dataset, 'json_files', None)
        val_files = getattr(cfg.dataset, 'val_json_files', None)
        train_datasets, test_datasets, _ = load_leaf_splits_datasets(
            root=cfg.dataset.root,
            train_files=train_files,
            val_files=val_files,
            holdout_files=None,
            limit=cfg.dataset.num_clients,
        )
            
            if uid in test_datasets:
                ted = test_datasets[uid]
                x_test, y_test = ted.tensors
                test_data[uid] = {'x': x_test.numpy(), 'y': y_test.numpy()}
            else:
                test_data[uid] = {'x': np.array([]), 'y': np.array([])}
                
        dataset = (users, groups, train_data, test_data)

    # call appropriate trainer
    t = optimizer(options, learner, dataset)
    t.train()

class DittoWrapper:
    def __init__(self, args, cfg=None):
        self.cfg = cfg
        
    def run(self):
        main(cfg=self.cfg)





