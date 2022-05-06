#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
from datetime import datetime
import argparse
import time
import numpy as np
import torch
import torch.utils.data as Data
from RegistrationDatasetForgradICON import dataset as RegistrationDataset
from utils import make_dir
from visualize_registration_results import show_current_images


# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# parse the commandline
parser = argparse.ArgumentParser()

# experiment setting parameters
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/PATH/TO/YOUR/DATA',
                    help="data path for training images")
parser.add_argument('-o','--output_path', required=True, type=str,
                        default=None,help='the path of output folder')
parser.add_argument('-e','--exp_name', required=True, type=str,
                        default=None,help='the name of the experiment')

# training parameters
parser.add_argument('-g','--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')



def prepare(args):
    output_path = args.output_path
    exp_name = args.exp_name
    data_path = args.datapath
    dataset_name = data_path.split('/')[-1]

    # Create experiment folder
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    exp_folder_path = os.path.join(output_path, dataset_name, exp_name, timestamp)
    make_dir(exp_folder_path)

    # Create checkpoint path, record path and log path
    checkpoint_path = os.path.join(exp_folder_path, "checkpoints")
    make_dir(checkpoint_path)
    record_path = os.path.join(exp_folder_path, "records")
    make_dir(record_path)
    log_path = os.path.join(exp_folder_path, "logs")
    make_dir(log_path)
    test_path = os.path.join(exp_folder_path, "tests")
    make_dir(test_path)

    torch.backends.cudnn.benchmark = True

    return exp_folder_path

def save_fig(save_path, fname, ite, phase, moving, target, warped, phi):
        """
        save 2d center slice from x,y, z axis, for moving, target, warped, l_moving (optional), l_target(optional), (l_warped)

        :param phase: train|val|test|debug
        :return:
        """
        visual_param = {}
        visual_param['visualize'] = False
        visual_param['save_fig'] = True
        visual_param['save_fig_path'] = save_path
        visual_param['save_fig_path_byname'] = os.path.join(save_path, 'byname')
        visual_param['save_fig_path_byiter'] = os.path.join(save_path, 'byiter')
        visual_param['save_fig_num'] = 4
        visual_param['pair_name'] = fname
        visual_param['iter'] = phase + "_iter_{:0>6d}".format(ite)

        show_current_images(ite, iS=moving, iT=target, iW=warped,
                            iSL=None, iTL=None, iWL=None,
                            vizImages=None, vizName="", phiWarped=phi,
                            visual_param=visual_param, extraImages=None, extraName="") 

def normalize(phi, shape):
    for i in range(len(shape)):
            phi[:, i, ...] = 2 * (phi[:, i, ...] / (shape[i] - 1) - 0.5)
    return phi

inshape = [175, 175, 175]

if __name__ == "__main__":
    args = parser.parse_args()
    exp_dir = prepare(args)
    model_dir = os.path.join(exp_dir, "checkpoints")
    figure_dir = os.path.join(exp_dir, "records")

    training_generator = Data.DataLoader(RegistrationDataset(args.datapath), batch_size=1,
                                                shuffle=True, num_workers=2)

    # extract shape from sampled input
     #next(training_generator)[0][0].shape[1:-1]

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    if args.load_model:
        # load initial model (if specified)
        model = vxm.networks.VxmDense.load(args.load_model, device)
    else:
        # otherwise configure new model
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=False,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize
        )

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    losses = [image_loss_func]
    weights = [1]

    # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [args.weight]

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):

        # save model checkpoint
        if epoch % 20 == 0:
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for X, Y, fname in training_generator:
            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            X = X.to(device).float()
            Y = Y.to(device).float()
            inputs = [X, Y]
            y_true = [Y, Y]

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

        #plot figure
        if epoch % 2 == 0:
            save_fig(figure_dir, fname, epoch, "train", X.cpu(), Y.detach().cpu(), y_pred[0].detach().cpu(), normalize(y_pred[-1].detach()+model.transformer.grid, [175, 175, 175]).cpu())

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
