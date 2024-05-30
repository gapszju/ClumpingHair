import logging
import os
import shutil
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class sNet(nn.Module):

    def __init__(self, base_model: str, out_dim: int, in_channels: list[int]):
        super(sNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # input conv layers
        self.conv1 = nn.Conv2d(in_channels[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        del self.backbone.conv1
        del self.backbone.bn1
        
        # add mlp projection head
        self.mlp = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        del self.backbone.fc

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise KeyError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        x = []
        
        if x1 is not None:
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = backbone.relu(x1)
            x1 = backbone.maxpool(x1)
            x.append(x1)
        
        if x2 is not None:
            x2 = self.conv2(x2)
            x2 = self.bn2(x2)
            x2 = backbone.relu(x2)
            x2 = backbone.maxpool(x2)
            x.append(x2)

        x = torch.cat(x, dim=0)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)

        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)

        return x


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.max_acc = 0
        self.writer = SummaryWriter(comment="_"+self.args.comment)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
    
    def info_nce_loss(self, features, params=None):
        batch_size = features.shape[0] // 2
        features = F.normalize(features, dim=-1)

        features1, features2 = features[:batch_size], features[batch_size:]
        similarity_matrix = torch.matmul(features1, features2.T)
        
        if params is None:
            penalty = 1
        else:
            param_diff_matrix = (params[:, None] - params[None, :])**2
            penalty = 1 / (1 + param_diff_matrix)

        logits = similarity_matrix / penalty / self.args.temperature
        labels = torch.arange(batch_size).to(self.args.device)
        
        return logits, labels
    
    def train(self, train_loader, test_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")

        for epoch_counter in trange(self.args.epochs):
            loss_am = AverageMeter()
            
            self.model.train()
            for images, params in tqdm(train_loader, leave=False):
                x1, x2 = images
                x1, x2 = x1.to(self.args.device), x2.to(self.args.device)
  
                param_img = x2[:, -1].float()
                params = param_img.sum(dim=(1, 2))
                params /= param_img.count_nonzero(dim=(1, 2)) + 1e-8
                
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(x1, x2)
                    logits, labels = self.info_nce_loss(features, params)
                    loss = self.criterion(logits, labels)
                    
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                loss_am.update(loss.item(), x1.size(0))

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
        
            if epoch_counter % self.args.log_every_n_epoch == 0:
                self.writer.add_scalar('train/loss', loss_am.avg, global_step=epoch_counter)
                
                torch.cuda.empty_cache()
                self.validation(train_loader, test_loader, epoch_counter)
                torch.cuda.empty_cache()

                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss_am.avg}\t")
            
            loss_am.reset()

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    @torch.no_grad()
    def validation(self, train_loader, test_loader, global_step):
        loss_am = AverageMeter()
        
        self.model.eval()
        
        # test loss
        for images, _ in test_loader:
            x1, x2 = images
            x1, x2 = x1.to(self.args.device), x2.to(self.args.device)

            with autocast(enabled=self.args.fp16_precision):
                features = self.model(x1, x2)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)
            loss_am.update(loss.item(), x1.size(0))
            
        self.writer.add_scalar('test/loss', loss_am.avg, global_step=global_step)
        
        # accuracy
        self.calc_accuracy(train_loader, global_step, label="train")
        top1, top5 = self.calc_accuracy(test_loader, global_step, label="test")
        
        # save best model
        if top1 > self.max_acc:
            self.max_acc = top1
            checkpoint_name = 'model_best.pth.tar'
            torch.save({
                'epoch': global_step,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(self.writer.log_dir, checkpoint_name))
            
    def calc_accuracy(self, data_loader, global_step, label="train"):
        top1_am, top5_am = AverageMeter(), AverageMeter()
        top1_p_am, top5_p_am = AverageMeter(), AverageMeter()
        dataset = data_loader.dataset
        for idx in range(dataset.n_hair):
            images, _ = dataset.get_val(idx)
            x1, x2 = images
            x1, x2 = x1.to(self.args.device), x2.to(self.args.device)

            with autocast(enabled=self.args.fp16_precision): 
                features = self.model(x1, x2)
                logits, labels = self.info_nce_loss(features)
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                
                top1_am.update(top1.item())
                top5_am.update(top5.item())
                
                # optional: patch-wise accuracy
                if dataset.crop_size:
                    N, P = dataset.n_classes, dataset.crop_size
                    patch = x1.unfold(2, P, 32).unfold(3, P, 32).reshape(N, 1, -1, P, P)
                    features1 = [self.model(patch[:, :, i], None) for i in range(patch.shape[2])]
                    patch = x2.unfold(2, P, 32).unfold(3, P, 32).reshape(N, 3, -1, P, P)
                    features2 = [self.model(None, patch[:, :, i]) for i in range(patch.shape[2])]
                    
                    logits, labels = [], None
                    for f1, f2 in zip(features1, features2):
                        _logits, labels = self.info_nce_loss(torch.cat([f1, f2]))
                        logits.append(_logits)
                    logits = torch.stack(logits).mean(dim=0)
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    top1_p_am.update(top1.item())
                    top5_p_am.update(top5.item())

        self.writer.add_scalar(f"{label}/acc/top1", top1_am.avg, global_step=global_step)
        self.writer.add_scalar(f"{label}/acc/top5", top5_am.avg, global_step=global_step)
        
        if dataset.crop_size:
            self.writer.add_scalar(f"{label}/acc_patch/top1", top1_p_am.avg, global_step=global_step)
            self.writer.add_scalar(f"{label}/acc_patch/top5", top5_p_am.avg, global_step=global_step)
            
            return top1_p_am.avg, top5_p_am.avg
        return top1_am.avg, top5_am.avg
    
    
class ParamRegression():
        
        def __init__(self, *args, **kwargs):
            self.args = kwargs['args']
            self.model = kwargs['model'].to(self.args.device)
            self.optimizer = kwargs['optimizer']
            self.scheduler = kwargs['scheduler']
            self.min_loss = 1e8
            self.writer = SummaryWriter(comment="_"+self.args.comment)
            logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
            self.criterion = torch.nn.MSELoss().to(self.args.device)
        
        def train(self, train_loader, test_loader):
    
            scaler = GradScaler(enabled=self.args.fp16_precision)
    
            logging.info(f"Start ParamRegression training for {self.args.epochs} epochs.")
    
            for epoch_counter in trange(self.args.epochs):
                loss_am = AverageMeter()
                
                self.model.train()
                for images, params in tqdm(train_loader, leave=False):
                    images = images.to(self.args.device)
                    params = params.float().to(self.args.device)
    
                    with autocast(enabled=self.args.fp16_precision):
                        logits = self.model(images, None).sigmoid().flatten()
                        loss = self.criterion(logits, params)
                        
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    loss_am.update(loss.item(), images.size(0))
    
                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()
            
                if epoch_counter % self.args.log_every_n_epoch == 0:
                    self.writer.add_scalar('train/loss', loss_am.avg, global_step=epoch_counter)
                    torch.cuda.empty_cache()
                    self.validation(test_loader, epoch_counter)
                    torch.cuda.empty_cache()
                    logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss_am.avg}\t")
                
                loss_am.reset()
            
            logging.info("Training has finished.")

        @torch.no_grad()
        def validation(self, test_loader, global_step):
            loss_am = AverageMeter()
            
            # test loss
            self.model.eval()
            for images, params in tqdm(test_loader, leave=False):
                images = images.to(self.args.device)
                params = params.float().to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(images, None).sigmoid().flatten()
                    loss = F.l1_loss(logits, params)
                    
                loss_am.update(loss.item(), images.size(0))
                
            self.writer.add_scalar('test/loss', loss_am.avg, global_step=global_step)
            
            # save best model
            if loss_am.avg < self.min_loss:
                self.min_loss = loss_am.avg
                checkpoint_name = 'model_best.pth.tar'
                torch.save({
                    'epoch': global_step,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, os.path.join(self.writer.log_dir, checkpoint_name))