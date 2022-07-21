#coding=utf-8
import os
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score
import copy

import utils.lr_decay as lrd
import utils.lr_sched as lr_sched
from utils.misc import NativeScalerWithGradNormCount
from net.mae_vit import ViT
from utils.loss import EmoCoT_loss

def parse_args():
    """
    Parsing configuration from the command line.    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home1/zrliu/dataset/abaw4/', help='Dataset path.')
    parser.add_argument('--init_batch_size', type=int, default=64, help='Initial batch size.')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers.')
    parser.add_argument('--epoch_num', type=int, default=6, help='Total training epochs.')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of expression categories.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Min lr after lr decay.')
    parser.add_argument('--blr', type=float, default=5e-4, help='Initial lr before linear zoom.')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='Epochs to warmup LR.')
    parser.add_argument('--accum_iter', type=int, default=4, help='Number of batches aggregated for one iteration.')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Drop path rate of ViT.')
    parser.add_argument('--layer_decay', type=float, default=0.65, help='Layer decay of ViT.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight_decay.')
    parser.add_argument('--clip_grad', type=float, default=0.05, help='Clip gradient.')
    parser.add_argument('--mask_rate', type=float, default=0, help='Mask rate of ViT patch in training.')
    parser.add_argument('--vit_num', type=int, default=2, help='Number of models used for co-training.')
    parser.add_argument('--repeat_count', type=int, default=1, help='Load data once and iterate parameters 1+ times.')
    parser.add_argument('--pretrained_model_paths', type=str, nargs='+', default=['./models/mae_norm.pth'], help='Checkpoint paths of pretrained models.')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPU ids.')
    parser.add_argument('--val_folders', type=str, nargs='+',default=['val'], help='Validation set folder name.')
    parser.add_argument('--idx', type=int, default=1, help='ID used to distinguish multiple experiments with the same configuration.')
    parser.add_argument('--checkpoint_path', type=str, default='./experiments', help='Path to save checkpoint.')
    parser.add_argument('--zoom_lr', action='store_true', help='Whether to zoom learning rate by batch size and accum iter.')
    parser.add_argument('--lambda_CE', type=int, default=1, help='Balance of JS loss and CE loss.')
    parser.add_argument('--lambda_JS', type=int, default=100, help='Balance of JS loss and CE loss.')
    parser.add_argument('--max_batch_size', type=int, default=4096, help='Max batch size.')
    args = parser.parse_args()
    
    # Zoom batch size by mask rate(a large mask rate can greatly reduced storage space consumption).
    args.batch_size = args.init_batch_size * int(1 / (1-args.mask_rate))
    if args.batch_size > args.max_batch_size:
        args.batch_size = args.max_batch_size
    if args.zoom_lr:    
        # Zoom learning rate by batch size and accum iter.
        args.lr = args.blr * args.batch_size * args.accum_iter / 256
    else:
        args.lr = args.blr
    
    # The number of pretrained model checkpoints should match the number of ViT for co-training.
    assert args.vit_num % len(args.pretrained_model_paths) == 0, \
        "Pretraining model cannot be allocated equally."

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    args.checkpoint_dir_name = gene_checkpoint_dir_name(args, parser)
    
    return args

def build_models(args):
    """
    Build ViT models for co-training.
    """
    
    models = []
    param_groups = []

    models_of_cpt = (args.vit_num // len(args.pretrained_model_paths))
    for i in range(args.vit_num):
        checkpoint_path = args.pretrained_model_paths[i // models_of_cpt]
        model = ViT(num_classes=args.num_classes, drop_path_rate = args.drop_path_rate, \
            finetune_path=checkpoint_path, myismask=True)
        model_without_ddp = model.vit
        param_groups += lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                            no_weight_decay_list=model_without_ddp.no_weight_decay(),
                            layer_decay=args.layer_decay
                        )    
        models.append(model)
    return models, param_groups
            
def train(models, param_groups, args):
    """Train and save model information.

    Args:
        models (list): list of ViT used for co-training
        param_groups (list): parameters groups of ViTs
        args : arguments parsed by parser
    """
    
    train_dataset, train_dataloader = prepare_data('train', 
                                                   args.data_path+'train', 
                                                   args.batch_size, 
                                                   args.workers)
    
    print('=> training set: {} images'.format(len(train_dataset)))
    
    val_datasets, val_dataloaders = [], []
    for val_folder in args.val_folders:
        val_dataset, val_dataloader = prepare_data('eval', 
                                                   args.data_path+val_folder, 
                                                   args.init_batch_size * 4, # for faster
                                                   args.workers)
        val_datasets.append(val_dataset)
        val_dataloaders.append(val_dataloader)
        
    loss_func = EmoCoT_loss(dist_num=args.vit_num, lambda_JS=args.lambda_JS, lambda_exp=args.lambda_CE) # co-training loss consisting of CE and JS
    optimizer = torch.optim.AdamW(param_groups, args.lr)    
    loss_scaler = NativeScalerWithGradNormCount()
    
    for epoch in range(args.epoch_num):
        acc, mean_f1, loss = train_one_epoch(epoch, models, loss_func, \
                                             loss_scaler, optimizer, train_dataloader, args)
        print("=> Epoch[{}] loss: {:.4f}, accuracy: {:.2f}%, F1: {:.2f}%.".format(epoch, \
                                                                                loss, acc, mean_f1))
        
        val_acc_arr, val_mean_f1_arr, val_f1_arr = [], [], []
        for i, val_folder in enumerate(args.val_folders):
            acc_, mean_f1_, f1_ = evaluate(models, val_dataloaders[i], args)
            print("=> {} accuracy: {:.2f}%, F1: {:.2f}%.".format(val_folder, acc_, mean_f1_))
            val_acc_arr.append(acc_)
            val_mean_f1_arr.append(mean_f1_)
            val_f1_arr.append(f1_)
        checkpoint = {
            'epoch' : epoch, 
            'args' : vars(args),
            'train_acc' : acc, 
            'train_mean_f1' : mean_f1, 
            'train_loss' : loss,
            'val_folders' : args.val_folders, 
            'val_acc' : val_acc_arr,
            'val_mean_f1' : val_mean_f1_arr, 
            'val_f1_arr' : val_f1_arr,
            'state_dicts' : [copy.copy(model.state_dict()) for model in models]
        }
        save_checkpoint(checkpoint, 
                        os.path.join(args.checkpoint_path, args.checkpoint_dir_name), 
                        'checkpoint_{}.pkl'.format(epoch))

def prepare_data(mode, data_path, batch_size, num_workers):
    """Prepare dataset and dataloader for train or eval.

    Args:
        mode (bool): 'train' or 'eval'
        data_path (str): path of data
        batch_size (int): batch size
        num_workers (int): the number of workers used to load data

    Returns:
        tuple: dataset and dataloader
    """
    assert mode in ['train', 'eval'], "mode should be 'train' or 'eval'."
    
    if mode == 'train':
        transform = transforms.Compose([
                transforms.Resize([int(224*1.02), int(224*1.02)]),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
    else:
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
            ]) 
    
    # use ImageFolder to load format data.    
    dataset = ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size = batch_size,
                                               num_workers = num_workers,
                                               shuffle = (mode == 'train'),  
                                               pin_memory = True)
    
    return dataset, dataloader

def train_one_epoch(epoch, models, loss_func, loss_scaler, optimizer, dataloader, args):
    """
    Train models one epoch.
    """
    for model in models:    model.train()
    optimizer.zero_grad()
    iter_cnt = 0
    predicts, targets = [], []
    for imgs, tgts in tqdm(dataloader, desc="Epoch[{}]".format(epoch)):
        imgs, tgts = imgs.cuda(), tgts.cuda()
        for rep_ in range(args.repeat_count):
            outs = [model((imgs, args.mask_rate)) for model in models]
            loss = loss_func(outs, tgts)
            if iter_cnt % args.accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, 
                    (iter_cnt+1) / (len(dataloader) * args.repeat_count) + epoch, 
                    args)
            loss /= args.accum_iter
            params = []
            for model in models:    params += list(model.parameters())
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=params, 
                        create_graph=False,
                        update_grad=((iter_cnt + 1) % args.accum_iter == 0))
            if (iter_cnt + 1) % args.accum_iter == 0:
                optimizer.zero_grad()
            
            preds = sum(outs).argmax(axis=1)
            predicts.extend(preds.detach().cpu().numpy())
            targets.extend(tgts.detach().cpu().numpy())

            iter_cnt += 1

    predicts, targets = np.array(predicts), np.array(targets)
    acc = (predicts == targets).sum() / len(predicts) * 100
    mean_f1, _ = compute_f1(predicts, targets, args.num_classes)
    
    return acc, mean_f1, loss.detach().cpu().item()

def evaluate(models, dataloader, args):
    """
    Evaluate the accuracy and F1 value of models on the validation set.
    """
    with torch.no_grad():
        predicts, targets = [], []
        for model in models: model.eval()
        for (imgs, tgts) in tqdm(dataloader):
            imgs, tgts = imgs.cuda(), tgts.cuda()
            outs = [model((imgs, 0)) for model in models]                    
            preds = sum(outs).argmax(axis=1)
            predicts.extend(preds.cpu().numpy())
            targets.extend(tgts.cpu().numpy())
        predicts, targets = np.array(predicts), np.array(targets)    
        mean_f1, f1 = compute_f1(predicts, targets, args.num_classes)   
        acc = (predicts == targets).sum() / len(predicts) * 100
    return acc, mean_f1, f1

def compute_f1(preds,targets,num_classes):
    """
    Compute F1 value of models' prediction.

    Args:
        preds (numpy.ndarray): a numpy array containing the prediction results of all samples.
        targets (numpy.ndarray): a numpy array containing the label of all samples.
        num_classes (int): number of expression categories.

    Returns:
        tuple: (mean f1, f1 array)
    """
    f1=[]
    for i in range(0,num_classes):
        f1.append(f1_score(preds==i, targets==i) * 100)
    return np.mean(f1), f1

def count_parameters(models):
    """
    Count the number of parameters that need to be trained in the model
    """
    num = 0
    for model in models:
        num += sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num

def save_checkpoint(state, checkpointDir, filename_):
    """
    Save checkpoint containing models information and state_dict.
    """
    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)
    filename = os.path.join(checkpointDir, filename_)
    torch.save(state, filename)
    print('=> {} is saved in {}'.format(filename_, checkpointDir))

def gene_checkpoint_dir_name(args, parser):
    """
    Generate a unique name using for saving checkpoint according to the configuration.
    """
    hyperparameters_simp = ['ibs', 'ep', 'nc', 'ml', 'blr', 'wp',  
                     'accum', 'dpr', 'ld', 'wd', 'cg', 'mr', 'vn', 'rc', 'zl', 'wCE', 'wJS', 'mbs']
    hyperparameters = [
        'init_batch_size', 'epoch_num', 'num_classes', 'min_lr', 
        'blr', 'warmup_epochs', 'accum_iter', 'drop_path_rate', 
        'layer_decay', 'weight_decay', 'clip_grad', 'mask_rate', 
        'vit_num', 'repeat_count', 'zoom_lr', 'lambda_CE', 'lambda_JS', 'max_batch_size'
    ]    
    
    checkpoint_dir_name = ""
    
    data_path = getattr(args, 'data_path')
    if data_path != parser.get_default('data_path'):
        split_path = data_path.split('/')
        while '' in split_path: split_path.remove('')
        dataset = split_path[-1]
        checkpoint_dir_name += '[{}]'.format(dataset)
    
    for param, param_simp in zip(hyperparameters, hyperparameters_simp):
        arg_val = getattr(args, param)
        if arg_val != parser.get_default(param):
            checkpoint_dir_name += "[{}{}]".format(param_simp, arg_val)
    
    if args.pretrained_model_paths != parser.get_default('pretrained_model_paths'):
        paths = [path.split('/')[-1][:-4] for path in args.pretrained_model_paths]
        checkpoint_dir_name += "["
        for path in set(paths):
            checkpoint_dir_name += "{}_{}".format(path, paths.count(path))
        checkpoint_dir_name += "]"
        
    if checkpoint_dir_name == "":
        checkpoint_dir_name = ("default_{}".format(args.idx))
    else:
        checkpoint_dir_name = (checkpoint_dir_name + "_{}".format(args.idx))
    
    return checkpoint_dir_name

if __name__=="__main__":
    
    args = parse_args()
    print(args)
    
    assert torch.cuda.is_available(), 'Cuda is not available.'
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    models, param_groups = build_models(args)
    print("=> model has %.2fM trainable parameters in total." % (count_parameters(models) / 1e6))
    
    if torch.cuda.device_count() > 1:
        print('=> the number of GPUs is greater than 1, DataParallel mode motivates.')
        for i in range(len(models)):
            models[i] = nn.DataParallel(models[i]).cuda()
        
    train(models, param_groups, args)
