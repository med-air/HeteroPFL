"""
federated training
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from nets.models import DigitModel
import logging
import random
import pandas as pd
import importlib
from utils.util import setup_logger, get_timestamp


def initialize(args, log):
    train_loaders, val_loaders, test_loaders = [], [], []
    
    if args.data == 'digits5':
        model = DigitModel()
        loss_fun = nn.CrossEntropyLoss()
        from dataset.get_datasets import get_digits5
        train_sites, val_sites, trainsets, valsets, testsets = get_digits5(args, log)

    else:
        raise ValueError('Unknown dataset: {}'.format(args.data))

    if args.debug:
        trainsets = [torch.utils.data.Subset(trset, list(range(args.batch*3))) for trset in trainsets]
        valsets = [torch.utils.data.Subset(trset, list(range(args.batch*2))) for trset in trainsets]
        testsets = [torch.utils.data.Subset(trset, list(range(args.batch*2))) for trset in trainsets]
    if args.subset < 1.0:
        log.info(f'Subsampling training sets to {args.subset*100}%')
        trainsets = [torch.utils.data.Subset(trset, list(range(round(len(trset)*args.subset)))) for trset in trainsets]
        for tr in trainsets:
            print((f'New Train len={len(tr)}'))

    if args.balance: 
        # trim data to the same len as the minimum one
        min_data_len = min([len(s) for s in trainsets])
        print(f'Balance training set, using {args.percent*100}% training data')
        trainsets = [torch.utils.data.Subset(trset, list(range(int(min_data_len*args.subset)))) for trset in trainsets]
        for idx in range(len(trainsets)):
            print(f' Train={len(trainsets[idx])}')
            
    for idx in range(len(trainsets)):
        train_loaders.append(torch.utils.data.DataLoader(trainsets[idx], batch_size=args.batch, shuffle=True, drop_last=True))
        val_loaders.append(torch.utils.data.DataLoader(valsets[idx], batch_size=args.batch, shuffle=False))
    for idx in range(len(testsets)):
        test_loaders.append(torch.utils.data.DataLoader(testsets[idx], batch_size=args.batch, shuffle=False))
        
        
    return model, loss_fun, train_sites, val_sites, trainsets, valsets, testsets, train_loaders, val_loaders, test_loaders

if __name__ == '__main__':
    os.environ['TORCH_HOME'] = '../../torchhome'
    from configs import set_configs, parse_exp_name
    args = set_configs()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    
    args.save_path = '../../experiments/checkpoint/{}/seed{}'.format(args.data, seed) 
    exp_folder = '{}_rounds{}_localE{}_lr{}_batch{}_N{}'.format(args.mode, args.rounds, args.local_epochs, args.lr, args.batch, args.clients) if args.exp is None else args.exp
    exp_folder = parse_exp_name(args, exp_folder)
    
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = args.save_path

    # wandb.config.update(args)
        # setup the logger
    log_path = args.save_path.replace('checkpoint', 'log')
    args.log_path = log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = setup_logger(
        f"{args.mode}-{get_timestamp()}",
        log_path,
        screen=True,
        tofile=True,
    )
    
    logger.propagate = False

    logger.info('=============== args ================')
    logger.info(str(args))


    generalize_sites = None
    server_model, loss_fun, train_sites, val_sites, train_sets, val_sets, test_sets, train_loaders, val_loaders, test_loaders = initialize(args,logger)

    param_size = 0
    for param in server_model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in server_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    logger.info('model size: {:.3f}MB'.format(size_all_mb))
    

    train_total_len = sum([len(tr_set) for tr_set in train_sets])
    client_weights = [len(tr_set)/train_total_len for tr_set in train_sets] if not args.pure_avg else [1/len(train_sets)]*len(train_sets)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    
    
    logger.info(str(('### Deive:',device)))
    logger.info('Training Clients:{}'.format(train_sites))
    logger.info(str(('Clients Weights:', client_weights)))
    

    # setup the summarywriter
    from torch.utils.tensorboard import SummaryWriter
    args.writer = SummaryWriter(log_path)
    
    # setup trainer
    trainer_module = importlib.import_module(f'federated_baselines.{args.mode}')
    TrainerClass = getattr(trainer_module, 'Trainer')
    trainer = TrainerClass(
        args,
        logger,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=client_weights,
        generalize_sites=generalize_sites,
    )
    

    trainer.best_changed = False
    trainer.early_stop = 20

    if args.resume:
        ckpts = sorted(os.listdir(SAVE_PATH), key=lambda x: int(x.split('_')[-1]))
        print(ckpts)
        checkpoint = torch.load(os.path.join(SAVE_PATH,ckpts[-1]))
        trainer.server_model.load_state_dict(checkpoint['server_model'])
        if args.pfl:
            for client_idx in range(trainer.client_num):
                trainer.client_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(trainer.client_num):
                trainer.client_models[client_idx].load_state_dict(checkpoint['server_model'])
        trainer.best_epoch, trainer.best_acc  = checkpoint['best_epoch'], checkpoint['best_acc']
        trainer.start_iter = int(checkpoint['a_iter']) + 1

        trainer.train_loss = pd.read_csv(os.path.join(args.log_path, "train_loss.csv"), index_col=0).to_dict()
        trainer.train_acc = pd.read_csv(os.path.join(args.log_path, "train_acc.csv"), index_col=0).to_dict()
        trainer.val_loss = pd.read_csv(os.path.join(args.log_path, "val_loss.csv"), index_col=0).to_dict()
        trainer.val_acc = pd.read_csv(os.path.join(args.log_path, "val_acc.csv"), index_col=0).to_dict()
        trainer.test_loss = pd.read_csv(os.path.join(args.log_path, "test_loss.csv"), index_col=0).to_dict()
        trainer.test_acc = pd.read_csv(os.path.join(args.log_path, "test_acc.csv"), index_col=0).to_dict()

        print('Resume training from epoch {}'.format(trainer.start_iter))
    else:
        # log the best for each model on all datasets
        trainer.best_epoch = 0
        trainer.best_acc = 0. # [0. for j in range(trainer.client_num)] 
        trainer.start_iter = 0
    
    trainer.start(train_loaders, val_loaders, test_loaders, loss_fun, SAVE_PATH, generalize_sites)
    logging.shutdown()
    
