import torch
import torch.nn.functional as F

        #[8,2,320,448]
def EPE(input_flow, target_flow, sparse=False, mean=True):#这个两个尺寸在调用该函数前都已经assert过了, 完全一样
    EPE_map = torch.norm(target_flow-input_flow,p=2,dim= 1)#对dim=1求2范数 #tensor [B,320,448]
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    #network_output is a list of flowx ;x = 2, 3, 4, 5, 6, size = [w,h], [w,h]
    #for i in range(len(network_output)):
        #print(network_output[i].size())

        #torch.Size([8, 2, 320, 448])
        #torch.Size([8, 2, 40, 56])
        #torch.Size([8, 2, 20, 28])
        #torch.Size([8, 2, 10, 14])
        #torch.Size([8, 2, 5, 7])

    def one_scale(output, target, sparse):#output 为多尺度光流, target为固定GT
        b, _, h, w = output.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))#GT为稀疏点就(pooling)去适应outputs
        else:
            target_scaled = F.interpolate(target, (h, w), mode='area')#否则target就下采样差值来适应outputs
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:#兼容单张test和batchsize train
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)#output is a list, compute dis beetween each outputs's item  and target
    return loss


def realEPE(output, target, sparse=False):#在epe的基础上同尺度处理
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)
