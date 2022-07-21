mae_path='/data/yfli/models/mae_norm.pth' # you need to change the mae pretrained model path
data_dir='/data/yfli/ABAW4/LSD/' # you need to change the data directory which includes the image folders (train/ and val/, respectvely)
checkpoint_path='./experiments/' # you can change the checkpoint directory

python Masked_CoTEX_main.py --pretrained_model_paths=$mae_path --data_path=$data_dir --checkpoint_path=$checkpoint_path