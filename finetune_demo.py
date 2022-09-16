import os
import argparse
import collections
import numpy as np
import importlib

import torch
import torch.optim as optim
import torch.nn.functional as F 

from config import Config
from data.demo_finetune_dataset import DemoFinetuneDataset
from loss.perceptual import PerceptualLoss
from loss.gan import GANLoss
from loss.attn_recon import AttnReconLoss
from util.misc import to_cuda
from util.visualization import tensor2pilimage
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer



def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/fashion_256.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')

    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int)
    parser.add_argument('--no_resume', action='store_true')
 

    parser.add_argument('--output_dir', type=str, default='./demo_finetune_output')
    parser.add_argument('--input_dir', type=str, default='./')
    parser.add_argument('--file_pairs', type=str, default='./demo.txt')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)

    opt = Config(args.config, args, is_train=False)
    opt.distributed=False
    opt.logdir = os.path.join(opt.checkpoints_dir, opt.name)
    opt.device = torch.cuda.current_device()
    # opt.local_rank = args.local_rank
    opt.num_iteration = 200

    # create a model
    net_G, net_D, net_G_ema, opt_G, opt_D, sch_G, sch_D \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_D, net_G_ema, \
                          opt_G, opt_D, sch_G, sch_D, \
                          None)

    current_epoch, current_iteration = trainer.load_checkpoint(opt, args.which_iter)
    net_G = trainer.net_G_ema.train()

    tp = []
    for name, parms in net_G.named_parameters():  
        parms.requires_grad = True
        tp.append(parms )


    data_loader = DemoFinetuneDataset(args.input_dir, opt.data, False)
    reference_list, skeleton_list = [], []
    with open(args.file_pairs, 'r') as fd:
        files = fd.readlines()
        for file in files:
            person, skeleton = file.replace('/n', '').split(',')
            reference_list.append(person)
            skeleton_list.append(skeleton)

    perceptual_loss = PerceptualLoss(
                network=opt.trainer.vgg_param.network,
                layers=opt.trainer.vgg_param.layers,
                num_scales=getattr(opt.trainer.vgg_param, 'num_scales', 1),
                ).to('cuda')

    file, crop_func = opt.trainer.face_crop_method.split('::')
    file = importlib.import_module(file)
    crop_func = getattr(file, crop_func)

    face_loss = PerceptualLoss(
                    network=opt.trainer.vgg_param.network,
                    layers=opt.trainer.vgg_param.layers,
                    num_scales=1,
                    ).to('cuda')
    con_loss = AttnReconLoss(opt.trainer.attn_weights).to('cuda')

    os.makedirs(args.output_dir, exist_ok=True)

    optimizer = optim.Adam(net_G.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)

    num = 0

    for reference_path, skeleton_path in zip(reference_list, skeleton_list):
        data = data_loader.load_item(reference_path, skeleton_path)
        data = to_cuda(data)


        if num < 1:
            reference_paths = list(set(reference_list))
            if len(reference_paths) >1:
                data = data_loader.load_item(reference_paths[0], reference_paths[0][:-4]+'.txt')
                data = to_cuda(data)
                input_image = data['reference_image']
                skeleton = torch.cat([data['source_skeleton'], data['source_skeleton']],1)


                for k in reference_paths[1:]:
                    data = data_loader.load_item(k, k[:-4]+'.txt')
                    data = to_cuda(data)
                    input_image1 = data['reference_image']
                    skeleton1 = torch.cat([data['source_skeleton'], data['source_skeleton']],1)
                    input_image = torch.cat([input_image, input_image1  ], 0)
                    skeleton  = torch.cat([skeleton, skeleton1], 0)
            else:
                input_image = data['reference_image']
                skeleton = torch.cat([data['source_skeleton'], data['source_skeleton']],1)

            print(skeleton.shape)

            for iters in range(opt.num_iteration+1):
                
                # print(input_image.shape, data['source_skeleton'].shape)
                output_dict = net_G(
                    input_image, 
                    # torch.cat([data['reference_image'], data['source_skeleton'][0]],1), 
                    skeleton, 
                )
                
                fake_image, info = output_dict['fake_image'], output_dict['info']

                p_loss = perceptual_loss(fake_image, input_image)
                f_loss = face_loss(crop_func(fake_image, data['face_center']), crop_func(input_image, data['face_center']))
                c_loss = con_loss(info, input_image)

                total_loss = opt.trainer.loss_weight.weight_face * f_loss + opt.trainer.loss_weight.weight_perceptual * p_loss \
                                + opt.trainer.loss_weight.weight_attn_rec * c_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


                if iters % 50 == 0:
                    
                    attn_image = info['transport_image']
                    # print(len(attn_image))
                    for i in range(len(attn_image)):
                        # print(attn_image[i].shape)
                        attn_image[i] = F.interpolate(attn_image[i], input_image.shape[2:] )
                    attn_image = torch.cat(attn_image, 3)

                    image = torch.cat([
                        input_image,
                        fake_image, skeleton[:,:3],  attn_image], 3).clip(-1, 1)

                    path = os.path.splitext(os.path.basename(reference_path))[0]+'_2_'+os.path.splitext(os.path.basename(reference_path))[0]
                    image = tensor2pilimage(image[0], minus1to1_normalized=True)
                    image.save("./{}/{}_{}.png".format(args.output_dir,path,str(iters)))
                    print("save image to ./{}/{}_{}.png".format(args.output_dir,path,str(iters)))
                    print("Perceptual loss:{:4f}; face loss:{:4f}; constraint:{:4f};".format(p_loss,f_loss,c_loss))
        else:
            input_image = data['reference_image']
            # print(input_image.shape, data['source_skeleton'].shape)
            output_dict = net_G(
                input_image, 
                # torch.cat([data['reference_image'], data['source_skeleton'][0]],1), 
                torch.cat([data['source_skeleton'], data['target_skeleton']],1), 
            )

            
            fake_image, info = output_dict['fake_image'], output_dict['info']
            attn_image = info['transport_image']
            for i in range(len(attn_image)):
                attn_image[i] = F.interpolate(attn_image[i], input_image.shape[2:] )
            attn_image = torch.cat(attn_image, 3)

            image = torch.cat([
                        data['reference_image'],
                        fake_image, data['target_skeleton'][:,:3],  attn_image], 3).clip(-1, 1)

            path = os.path.splitext(os.path.basename(reference_path))[0]+'_2_'+os.path.splitext(os.path.basename(skeleton_path ))[0]
            image = tensor2pilimage(image[0], minus1to1_normalized=True)
            image.save("./{}/{}.png".format(args.output_dir,path))

        num += 1



