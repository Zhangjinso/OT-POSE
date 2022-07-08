import math
import functools

import torch.nn as nn
from torch.nn import functional as F

from generators.base_function import EncoderLayer, DecoderLayer, ToRGB
from generators.my_ot import  sinkhorn

import torch

class Encoder(nn.Module):
    def __init__(
        self, 
        size, 
        input_dim, 
        channels, 
        num_labels=None, 
        match_kernels=None, 
        blur_kernel=[1, 3, 3, 1], 
        ):
        super().__init__()
        self.first = EncoderLayer(input_dim, channels[size], 1)
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2))
        self.log_size = log_size

        in_channel = channels[size]
        for i in range(log_size-1, 3, -1):
            out_channel = channels[2 ** i]
            num_label = num_labels[2 ** i] if num_labels is not None else None
            match_kernel = match_kernels[2 ** i] if match_kernels is not None else None
            use_extraction = num_label and match_kernel
            conv = EncoderLayer(
                in_channel, 
                out_channel, 
                kernel_size=3, 
                downsample=True, 
                blur_kernel=blur_kernel,
                use_extraction=use_extraction,
                num_label=num_label,
                match_kernel=match_kernel
                )

            self.convs.append(conv)
            in_channel = out_channel

    def forward(self, input, recoder=None):
        out = self.first(input)
        for layer in self.convs:
            out = layer(out, recoder)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        size,
        channels,
        num_labels,
        match_kernels,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()


        self.convs = nn.ModuleList()
        self.ot = nn.ModuleList()
        # input at resolution 16*16
        in_channel = channels[16]
        self.log_size = int(math.log(size, 2))
        self.embed = []
        
        for i in range(5, self.log_size + 1):
            out_channel = channels[2 ** i]
            num_label, match_kernel = num_labels[2 ** i], match_kernels[2 ** i]
            use_distribution = num_label and match_kernel
            upsample = (i != 4)
            
            base_layer = functools.partial(
                DecoderLayer,
                out_channel=out_channel,
                kernel_size=3, 
                blur_kernel=blur_kernel,
                use_distribution=use_distribution,
                num_label=num_label,
                match_kernel=match_kernel
                )
            up = nn.Module()   
            up.conv0 = base_layer(in_channel=in_channel, upsample=upsample)
            up.conv0_ = base_layer(in_channel=in_channel, upsample=upsample)
            up.conv1 = base_layer(in_channel=out_channel, upsample=False)
            up.to_rgb = ToRGB(out_channel, upsample=upsample)
            self.convs.append(up)
            in_channel = out_channel
                
        self.num_labels, self.match_kernels = num_labels, match_kernels
    
    
    def forward(self, input, neural_textures, recoder, src_image):
        counter = 0
        out, skip = input, None
        
        
        for i, up in enumerate(self.convs):
            
            res = out
            if self.num_labels[2**(i+4)] and self.match_kernels[2**(i+4)]:
                neural_texture_conv0 = neural_textures[counter]
                counter += 1    
            else:
                neural_texture_conv0, neural_texture_conv1 = None, None
            if self.match_kernels[2**(i+4)]:
                out,  tr_image = sinkhorn(out, neural_texture_conv0, image=src_image) 

                recoder['transport_image'].insert(0,tr_image)
            out = up.conv0(out)
            res = up.conv0_(res)
            out = up.conv1(out + res)
            skip = up.to_rgb(out, skip)
        image = skip
        return image
