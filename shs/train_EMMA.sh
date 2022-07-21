gpu_id=3
mae_path='/data/yfli/models/mae_norm.pth' # you need to change the mae pretrained model path
exp_path='/data/yfli/models/affecnet8_epoch5_acc0.6209.pth' # you need to change the AffectNet pretrained DAN model path
data_dir='/data/yfli/ABAW4/' # you need to change the data directory which includes the annotations (train and val) and cropped images folder

CUDA_VISIBLE_DEVICES=$gpu_id python EMMA_main.py --mae_pretrained_model_paths=$mae_path --exp_pretrained_model_paths=$exp_path --data_dir=$data_dir