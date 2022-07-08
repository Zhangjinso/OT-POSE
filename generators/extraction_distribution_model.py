import collections
from torch import nn
import sys
from generators.base_module import Encoder, Decoder

class Generator(nn.Module):
    def __init__(
        self,
        size,
        semantic_dim,
        channels,
        num_labels,
        match_kernels,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.size = size
        self.reference_encoder = Encoder(
            size, 3, channels, num_labels, match_kernels, blur_kernel
        )
            
        self.skeleton_encoder = Encoder(
            size, semantic_dim, channels, 
            )

        self.target_image_renderer = Decoder(
            size, channels, num_labels, match_kernels, blur_kernel
        )

    def _cal_temp(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def forward(
        self,
        source_image,
        skeleton,
    ):
        output_dict={}
        recoder = collections.defaultdict(list)
        skeleton_feature = self.skeleton_encoder(skeleton)
        
        _ = self.reference_encoder(source_image, recoder)
        neural_textures = recoder['encoder_output']
        

        output_dict['fake_image'] = self.target_image_renderer(
            skeleton_feature, neural_textures, recoder, source_image
            )
        output_dict['info'] = recoder
        return output_dict


if __name__ == "__main__":
    import torch

    a = Generator(256, 40, {16:512, 32:512, 64:512, 128:256, 256:128,512:64,1024:32}, {16:16,32:32, 64:32,128:64,256:False},{16:1,32:3,64:3,128:False,256:False})
    inp1 = torch.ones([2,3,256,176])
    inp2 = torch.ones([2,40,256,176])
    out = a(inp1, inp2)
    print(out['fake_image'].shape)