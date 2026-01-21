# optimizers.py
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def define_optimizer(opt, model):
    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    elif opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError(f'optimizer {opt.optimizer_type} not implemented')
    return optimizer

def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        raise NotImplementedError(f'learning rate policy {opt.lr_policy} not implemented')
    return scheduler