#coding=utf-8
import os
from itsdangerous import json
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.EmotionDataset import ABAWMTDataset

import utils.lr_decay as lrd
import utils.lr_sched as lr_sched
from utils.misc import NativeScalerWithGradNormCount
from net.EMMA import EMMA
from utils.loss import SingleMT_loss
from utils.metrics import all_metrics

def parse_args():
    """
    Parsing configuration from the command line.    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--blr', type = float, default = 5e-4)
    parser.add_argument('--min_lr', type = float, default = 1e-6)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epoch_num', type=int, default=30)
    
    parser.add_argument('--accum_iter', type=int, default=4)
    parser.add_argument('--drop_path_rate', type=int, default=0.1)
    parser.add_argument('--layer_decay', type=int, default=0.65)
    parser.add_argument('--weight_decay', type=int, default=0.05)
    parser.add_argument('--clip_grad', type=int, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                            help='epochs to warmup LR')

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--gpu', type=str, default='1',
                            help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading')
    parser.add_argument('--save_checkpoint', type=bool, default=False, help='whether to save checkpoint of the model')
    parser.add_argument('--mae_pretrained_model_paths', type=str, nargs='+', default='/data/yfli/models/mae_norm.pth', help='Checkpoint paths of MAE for ViT.')
    parser.add_argument('--exp_pretrained_model_paths', type=str, nargs='+', default='/data/yfli/models/affecnet8_epoch5_acc0.6209.pth', help='Checkpoint paths of DAN for CNN.')
    parser.add_argument('--data_dir', type = str, help = 'dataset direcory', default = '/data/yfli/ABAW4/')
    
    parser.add_argument('--log_dir', type = str, help = 'dir to save result txt files', default = 'logs/')
    parser.add_argument('--model_name', type = str, help = 'the name of models', default = 'EMMA')
    parser.add_argument('--middle_name', type = str, help = 'the supplement name of models', default = '')

    args = parser.parse_args()
    
    eff_batch_size = args.batch_size * args.accum_iter
    args.lr = args.blr * eff_batch_size/ 256
    
    if args.middle_name!='':    
        result_name = args.model_name + '_{model}_ABAW_MTL/'.format(model=args.middle_name)
    else:
        result_name = args.model_name + '_ABAW_MTL/'
        
    result_root = args.log_dir + result_name
    args.checkpoint_root = args.log_dir + result_name + 'checkpoints/'
    if not os.path.exists(result_root):
        os.makedirs(result_root)
        os.mkdir(args.checkpoint_root)
    args.log_file = result_root + 'log.txt'
    args.json_file = result_root + 'params_results.json'

    return args

def preparation(args):
    """
    Build EMMA model, optimizer, loss fun and dataset.
    """
    model = EMMA(num_classes=(8+12), drop_path_rate = args.drop_path_rate, finetune_path = args.mae_pretrained_model_paths, exp_model_path = args.exp_pretrained_model_paths)
    model.cuda()
    param_groups = lrd.param_groups_lrd(model.vit, args.weight_decay,
        no_weight_decay_list=model.vit.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer_vit = optim.AdamW(
        param_groups, args.lr
    )
    
    model_cnn = model.cnn
    for _, p in model_cnn.named_parameters():
        p.requires_grad = False
    optimizer_va = optim.Adam(model.va_head.parameters(), lr = 1e-4)
    
    loss_scaler = NativeScalerWithGradNormCount()
    loss_train_func = SingleMT_loss(return_losses=False)
    loss_test_func = SingleMT_loss()
    
    img_size = 224
    transform_train = transforms.Compose([
        transforms.Resize([int(img_size*1.04), int(img_size*1.04)]),
        ######augmentation######
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.2),
        transforms.RandomCrop([img_size, img_size]),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        ##################
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])    
    train_file = args.data_dir + '/training_set_annotations.txt'
    val_file = args.data_dir + '/validation_set_annotations.txt'
    data_root = args.data_dir + '/cropped_aligned/'    
    
    train_data = ABAWMTDataset(annotation_path=train_file, transform=transform_train, data_root=data_root)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    val_data = ABAWMTDataset(annotation_path=val_file, transform=transform_val, data_root=data_root)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    return model, optimizer_vit, optimizer_va, loss_scaler, loss_train_func, loss_test_func, train_loader, val_loader

def save_model(args, model, epoch, score, task):
    """
    Save model by tasks and clear the previous one
    """
    checkpoint_root_task = args.checkpoint_root + '/%s/'%task
    if not os.path.exists(checkpoint_root_task):
        os.makedirs(checkpoint_root_task)
    try:
        model_dir = os.listdir(checkpoint_root_task)
        if model_dir != []:
            for weight in model_dir:
                os.remove(checkpoint_root_task + weight)
    except:
        print('The file cannot be removed!')
    PATH = checkpoint_root_task + '{model}_{task}_{score}'.format(model=args.model_name, task=task, score=str(score) + '.pth')
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                }, PATH)
    print("epoch %d/%d model has been saved~\n"%(epoch, args.epoch_num))

def train_one_epoch(args, epoch, model, optimizer, optimizer_va, loss_scaler, loss_func, dataloader):
    """
    Train model for one epoch.
    """
    iter_cnt = 0
    train_loss = 0   
    iter_nums = len(dataloader)
    f= open(args.log_file, 'a+')
    model.train()
    for i, (batch_image, batch_label) in enumerate(dataloader):                
        batch_image, batch_label = batch_image.cuda(), batch_label.float().cuda()

        output = model(batch_image) 
            
        loss = loss_func(output, batch_label)
        
        if i % args.accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, i/iter_nums + epoch, args)
        
        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(i + 1) % args.accum_iter == 0)
        optimizer_va.step()
        
        if (i + 1) % args.accum_iter == 0:
            optimizer.zero_grad()
            optimizer_va.zero_grad()

        train_loss += loss

        if iter_cnt % args.print_freq == 0:
            print('%s: epoch %d/%d iter %d, train loss is %f.' % (args.model_name+'_'+args.middle_name,
                epoch, args.epoch_num, iter_cnt, loss.cpu()))
        iter_cnt += 1
            
    print("=====================") 
    print("epoch: {}".format(epoch)) 
    print('{:s} train loss: {:.8f}'.format(args.model_name+'_'+args.middle_name, train_loss)) 

    f.write("=====================") 
    f.write("epoch: {}/{}".format(epoch, args.epoch_num)) 
    f.write('  {:s} Train loss: {:.8f}'.format(args.model_name+'_'+args.middle_name, train_loss)) 

def val_one_epoch(args, model, val_loader, loss_func):
    """
    Evaluate the performance of the model
    """
    model.eval()
    all_test_loss = 0 
    with torch.no_grad():
        for i, (val_imgs, gt) in enumerate(val_loader):
            
            output = model(val_imgs.cuda())
            
            test_loss = loss_func(output, gt.float().cuda())
            all_test_loss += test_loss

            if i==0:
                result_all = output.cpu().numpy()
                gt_all = gt.cpu().numpy()
            else:
                result_all = np.concatenate((result_all, output.cpu().numpy()))
                gt_all = np.concatenate((gt_all, gt.cpu().numpy()))

    total_score, au_f1_av, VA_av, exp_f1, exp_acc = all_metrics(result_all, gt_all)

    f= open(args.log_file, 'a+')
    f.write("   {:s} test loss is {:.8f}, total score is {:.4f}, au_f1_av: {:.4f}, VA_ccc_av: {:.4f}, exp_f1_av: {:.4f}, exp_acc: {:.4f}".format(args.model_name, all_test_loss, total_score, au_f1_av, VA_av, exp_f1, exp_acc))
    f.write("=====================\n")
    print("{:s} test loss is {:.8f}, total score is {:.4f}\n, au_f1_av: {:.4f}, VA_ccc_av: {:.4f}, exp_f1_av: {:.4f}, exp_acc: {:.4f}".format(args.model_name, all_test_loss, total_score, au_f1_av, VA_av, exp_f1, exp_acc))
    print("=====================\n")
    return total_score, au_f1_av, VA_av, exp_f1

def EMMA_main():
    start_epoch = 1
    au_f1_max, VA_max, exp_max = 0.524, 0.46, 0.33
    total_score_max = 1.25
    args = parse_args()
    model, optimizer_vit, optimizer_va, loss_scaler, loss_train_func, loss_test_func, train_loader, val_loader = preparation(args)
    print("==> Start training")
    
    for epoch in range(start_epoch, args.epoch_num + 1):
        train_one_epoch(args, epoch, model, optimizer_vit, optimizer_va, loss_scaler, loss_train_func, train_loader)
        total_score, au_f1_av, VA_av, exp_f1 = val_one_epoch(args, model, val_loader, loss_test_func)

        if args.save_checkpoint:
            if au_f1_av>au_f1_max:
                au_f1_max = au_f1_av
                save_model(epoch, au_f1_av, task='AU')
            if VA_av>VA_max:
                VA_max = au_f1_av
                save_model(epoch, VA_av, task='VA')
            if exp_f1>exp_max:
                exp_max = exp_f1
                save_model(epoch, exp_f1, task='EXPR')
            if total_score > total_score_max:
                total_score_max = total_score
                save_model(epoch, total_score, task='MTL')
    
    result_dict = vars(args)
    json_file = args.json_file
    with open(json_file, 'w') as out:
        json.dump(result_dict, out, sorted_keys=False, indent=4)


if __name__=="__main__":
    EMMA_main()
