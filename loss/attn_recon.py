import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnReconLoss(nn.Module):
    def __init__(self,  weights={8:1, 16:0.5, 32:0.25, 64:0.125, 128:0.0625, 256:1/32, 512:1/64},):
        super(AttnReconLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.weights = weights
    
    def forward(self, attn_dict, gt_image):
        # softmax, query = attn_dict['extraction_softmax'], attn_dict['semantic_distribution']
        warped_image = attn_dict['transport_image']
        if isinstance(warped_image, list) :
            loss, weights = 0, 0
            for item_softmax in warped_image:
                # print(item_softmax.shape)
                h, w = item_softmax.shape[2:]
                gt_ = F.interpolate(gt_image, (h,w)).detach()
                
                loss += self.l1loss(item_softmax, gt_) * self.weights[h]
                weights += self.weights[h]
            loss = loss/weights
        else:
            h, w = warped_image.shape[2:]
            gt_ = F.interpolate(gt_image, (h,w))
            loss = self.l1loss(warped_image, gt_)
        return loss

    def cal_attn_image(self, input_image, softmax, query):
        b, num_label, h, w = query.shape
        if b != input_image.shape[0]:
            ib,ic,ih,iw = input_image.shape
            num_load_img = b // ib
            input_image = input_image[:,None].expand(ib, num_load_img, ic, ih, iw).contiguous()

        input_image = input_image.view(b, -1, h*w)
        extracted = torch.einsum('bkm,bvm->bvk', softmax, input_image)
        query = F.softmax(query.view(b, num_label, -1), 1)
        estimated_target = torch.einsum('bkm,bvk->bvm', query, extracted)
        estimated_target = estimated_target.view(b, -1, h, w)
        return estimated_target


        