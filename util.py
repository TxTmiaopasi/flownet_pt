import os
import numpy as np
import shutil
import torch


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        src = os.path.join(save_path,filename)
        dst = os.path.join(save_path,'model_best.pth.tar')
        shutil.copyfile(src,dst)#把当前最好的覆盖写


class AverageMeter(object):
    """Computes and stores the average and current value"""

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

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def flow2rgb(flow_map, max_value):# [2,370,1224]
    flow_map_np = flow_map.detach().cpu().numpy()#??what trick
    #eturned Tensor shares the same storage with the original one.
    # In-place modifications on either of them will be seen, and may trigger errors in correctness checks.

    _, h, w = flow_map_np.shape#[2,h,w]

    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')#??? 两幅图中某个位置像素都等于0的置位nan

    rgb_map = np.ones((3,h,w)).astype(np.float32)#占位符
    #normalization
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        #normalized_flow_map = (flow_map_np-flow_map_np.mean())/np.ndarray.std(flow_map_np)
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    #vector2color coding
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)#上溢,下溢处理,smaller than 0 become 0, and values larger than 1 become 1, 区间内的值不动

