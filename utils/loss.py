import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class EmoCoT_loss(nn.Module):
    def __init__(self, dist_num=2, lambda_JS=100, lambda_exp = 1):

        super(EmoCoT_loss, self).__init__()

        self.EXPlossfunc = nn.CrossEntropyLoss(reduction='mean')

        self.JSDiv = GeneralizedJSDiv(dist_num=dist_num)
        self.lambda_exp = lambda_exp 
        self.lambda_JS = lambda_JS
        self.lamd_coeff = 0.5
        self.eps = 1e-4

    def forward(self, preds, targets):
        loss = 0
        
        #######CE#######
        mask1 = targets.eq(-1)
        mask = ~mask1
        loss_exp = 0
        for pre in preds:
            loss_exp += self.EXPlossfunc(pre[mask, :], targets[mask].long())
        loss_exp /= len(preds)
        loss = loss + loss_exp * self.lambda_exp 
        ################
        
        #######JS#######
        JSloss = self.JSDiv(preds)
        loss = loss + JSloss * self.lambda_JS
        ################
        
        return loss

class GeneralizedJSDiv(torch.nn.Module):
    def __init__(self, dist_num=2, weights=None):
        super(GeneralizedJSDiv, self).__init__()
        self.eps = 1e-4
        self.dist_num = dist_num
        if weights:
            self.weights = [float(w)/dist_num for w in weights]
        else:
            w = 1/self.dist_num
            self.weights = [w] * self.dist_num
        
    def kl_div(self, target, prediction):
        output_pos = target * (target.clamp(min=self.eps).log() - prediction.clamp(min=self.eps).log())
        zeros = torch.zeros_like(output_pos)
        output = torch.where(target>0, output_pos, zeros)
        return output.mean()
    
    def forward(self, pred): #pred is a list contain N distributions
        preds = []
        for p in pred:
            preds.append(F.softmax(p, dim=1))

        mean_dist = sum([w*p for w, p in zip(self.weights, preds)])
        mean_dist = mean_dist.clamp(min=self.eps)

        loss = sum([w*self.kl_div(p, mean_dist) for w, p in zip(self.weights, preds)])
        return loss

class ABAWMT_BCE_sigmoid(nn.Module):
    def __init__(self, size_average=True, weight=None):
        super(ABAWMT_BCE_sigmoid, self).__init__()
        self.size_average = size_average
        if weight:
            self.weight = weight
        else:
            self.weight = [7.1008924, 15.63964869, 5.47108051, 2.80360066, 1.5152332, 1.89083564, 3.04637044, 34.04600245, 32.47861156, 36.76637801, 0.58118674, 11.1586486]
            self.weight = torch.tensor(self.weight).float().cuda()

    def forward(self, x, labels):
        N = x.size(0)
        mask1 = labels.eq(-1)

        mask = 1 - mask1.float()

        target = labels.gt(0)
        target = target.float()

        loss = F.binary_cross_entropy_with_logits(x, target, mask, pos_weight=self.weight, reduction='sum')

        if self.size_average:
            loss = loss / N

        return loss

class CCCLoss(nn.Module):
    def __init__(self, digitize_num=20, range=[-1, 1], weight=None):
        super(CCCLoss, self).__init__() 
        self.digitize_num =  digitize_num
        self.range = range
        self.weight = weight
        self.eps = 1e-4
        if self.digitize_num >1:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = torch.as_tensor(bins, dtype = torch.float32).cuda().view((1, -1))
    def forward(self, x, y): 
        y = y.float().view(-1)
        mask = ~(y==-5)
        new_x, new_y = x[mask], y[mask]
        
        if self.digitize_num !=1:
            new_x = F.softmax(new_x, dim=-1)
            new_x = (self.bins * new_x).sum(-1) # expectation
        new_x = new_x.view(-1)
        vx = new_x - torch.mean(new_x) 
        vy = new_y - torch.mean(new_y) 
        rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
        x_m = torch.mean(new_x)
        y_m = torch.mean(new_y)
        x_s = torch.std(new_x)
        y_s = torch.std(new_y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + self.eps)

        cccloss = 1-ccc
        return cccloss
    
class SingleMT_loss(nn.Module):
    def __init__(self, lambda_VA=1, lambda_AU = 1, lambda_exp = 1, return_losses=False):

        super(SingleMT_loss, self).__init__()

        self.AUlossfunc = ABAWMT_BCE_sigmoid(size_average=True) 
        self.VAlossfunc = CCCLoss(digitize_num=1)
        self.explossfunc = nn.CrossEntropyLoss(reduction='none')

        self.BCE = nn.BCELoss() 
        self.sigmoid = nn.Sigmoid() 
        self.log_sigmoid = nn.LogSigmoid() 
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self.lambda_AU = lambda_AU
        self.lambda_VA = lambda_VA 
        self.lambda_exp = lambda_exp 
        self.eps = 1e-4
        self.return_losses = return_losses
    
    def forward(self, outputs, targets):
        VA_output, exp_output, AU_output = outputs[:, :2], outputs[:, 2:10], outputs[:, 10:]
        VA_label, exp_label, AU_label = targets[:, :2], targets[:, 2], targets[:, 3:]
        loss = 0
        #######VA#######
        loss_VA = self.VAlossfunc(VA_output[:,0], VA_label[:,0]) + self.VAlossfunc(VA_output[:,1], VA_label[:,1])
        loss_VA = loss_VA * self.lambda_VA
        loss = loss + loss_VA 
        ################
        #######Exp######
        mask1 = exp_label.eq(-1)
        mask = ~mask1
        exp_label[mask1] = 0
        loss_exp = self.explossfunc(exp_output, exp_label.long())
        loss_exp = loss_exp[mask].mean() * self.lambda_exp 
        loss = loss + loss_exp
        ################
        #######AU#######
        loss_AU = self.AUlossfunc(AU_output, AU_label) * self.lambda_AU 
        loss = loss + loss_AU 
        ################
        
        if self.return_losses:
            return loss_VA, loss_AU, loss_exp
        return loss
    