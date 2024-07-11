import os
import time
import errno
import torch
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(test_data,labels,model,criterion, device, debug_='MEDIUM',batch_size=64, isAdvReg=0):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time() 
    
    len_t = len(test_data)//batch_size
    if len(test_data)%batch_size:
        len_t += 1

    data_time.update(time.time() - end)
 
    total = 0
    for ind in range(len_t):
        inputs =  test_data[ind*batch_size:(ind+1)*batch_size].to(device)
        targets = labels[ind*batch_size:(ind+1)*batch_size].to(device)

        total += len(inputs) 
        # compute output
        # compute output
        outputs = model(inputs)

        if(type(outputs)==tuple):
            outputs = outputs[0]
        

        loss = criterion(outputs, targets) 
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

 


    return (losses.avg, top1.avg)

def save_checkpoint_user(user_num, state, is_best, checkpoint=None, filename='checkpoint.pth.tar',extra_checkpoints=False):
    assert (checkpoint != None), 'Error: No checkpoint path provided!'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    if extra_checkpoints:
        e_path=checkpoint+'/user_%d_checkpoints'%user_num
        if not os.path.isdir(e_path):
            mkdir_p(e_path)
        e_filepath=e_path+'/checkpoint_epoch_%d.pth.tar'%state['epoch']
        print('User %d saving extra checkpoint @epoch %d'%(user_num,state['epoch']))
        torch.save(state, e_filepath)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'user_%d_model_best.pth.tar'%(user_num)))

def save_checkpoint_global(state, is_best, checkpoint=None, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    assert (checkpoint != None), 'Error: No checkpoint path provided!'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    #torch.save(state, filepath)
    if is_best:
        filepath = os.path.join(checkpoint, best_filename)
        torch.save(state, filepath)
        #shutil.copyfile(filepath, os.path.join(checkpoint, best_filename))

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise