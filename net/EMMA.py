import torch
import torch.nn as nn
from net.dan import DAN
from net.mae_vit import ViT

class EMMA(ViT):
    def __init__(self, exp_model_path, **kwargs):
        super().__init__(**kwargs)
        self.cnn = DAN(num_class=8)
        model_parameters = torch.load(exp_model_path)
        self.cnn.load_state_dict(model_parameters['model_state_dict'])
        print("Expression pretrained model has been loaded!!!")
        
        self.cnn_num_features = 512
        self.vit_num_features = self.vit.num_features
        self.hidden_features = 512
        self.vit.head = nn.Sequential(
            # nn.BatchNorm1d()
            nn.Linear(self.vit_num_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.num_classes)
        )
        va_num_features = self.num_classes + 8
        va_hidden_features = 256
        self.va_head = nn.Sequential(
            nn.Linear(va_num_features, va_hidden_features),
            nn.ReLU(),
            nn.Linear(va_hidden_features, 2)
        )
            
    def forward(self, x):
        vit_outputs = self.vit(x)
        cnn_outputs, _, heads = self.cnn(x)

        concat_outputs = torch.cat([cnn_outputs, vit_outputs], dim=1)
        
        va_outputs = self.va_head(concat_outputs)
        
        outputs = torch.cat([va_outputs, vit_outputs], dim=1)
        return outputs
