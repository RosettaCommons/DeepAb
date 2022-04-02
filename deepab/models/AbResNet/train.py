import argparse
import os
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import deepab
from deepab.models.AbResNet import AbResNet, load_model
from deepab.models.PairedSeqLSTM import load_model as load_lstm_rnn
from deepab.util.masking import MASK_VALUE
from deepab.util.training import check_for_h5_file
# from deepab.util.util import RawTextArgumentDefaultsHelpFormatter
from deepab.datasets.H5PairwiseGeometryDataset import H5PairwiseGeometryDataset
from deepab.preprocess.generate_h5_pairwise_geom_file import antibody_to_h5

_output_names = ['ca_dist', 'cb_dist', 'no_dist', 'omega', 'theta', 'phi']


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,
                 weight=None,
                 gamma=2,
                 reduction='mean',
                 ignore_index=MASK_VALUE):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input,
                                  target,
                                  reduction=self.reduction,
                                  weight=self.weight,
                                  ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt)**self.gamma * ce_loss).mean()
        return focal_loss


def train_epoch(model, train_loader, optimizer, device, criterion, loss_size):
    """Trains a model for one epoch"""
    model.train()
    running_losses = torch.zeros(loss_size)
    for inputs, labels in tqdm(train_loader, total=len(train_loader)):
        inputs = inputs.to(device)
        labels = [label.to(device) for label in labels]

        optimizer.zero_grad()

        def handle_batch():
            outputs = model(inputs)

            losses = [
                criterion(output, label)
                for output, label in zip(outputs, labels)
            ]
            total_loss = sum(losses)
            losses.append(total_loss)

            total_loss.backward()
            optimizer.step()
            return outputs, torch.Tensor([float(l.item()) for l in losses])

        outputs, batch_loss = handle_batch()
        running_losses += batch_loss

    return running_losses


def validate(model, validation_loader, device, criterion, loss_size):
    """"""
    with torch.no_grad():
        model.eval()
        running_losses = torch.zeros(loss_size)
        for inputs, labels in tqdm(validation_loader,
                                   total=len(validation_loader)):
            inputs = inputs.to(device)
            labels = [label.to(device) for label in labels]

            def handle_batch():
                outputs = model(inputs)

                losses = [
                    criterion(output, label)
                    for output, label in zip(outputs, labels)
                ]
                total_loss = sum(losses)
                losses.append(total_loss)

                return outputs, torch.Tensor([float(l.item()) for l in losses])

            outputs, batch_loss = handle_batch()
            running_losses += batch_loss
    return running_losses


def train(model,
          train_loader,
          validation_loader,
          optimizer,
          epochs,
          current_epoch,
          device,
          criterion,
          lr_modifier,
          writer,
          save_file,
          save_every,
          properties=None):
    """"""
    properties = {} if properties is None else properties
    print('Using {} as device'.format(str(device).upper()))
    model = model.to(device)
    loss_size = len(_output_names) + 1

    for epoch in range(current_epoch, epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device,
                                   criterion, loss_size)
        avg_train_losses = train_losses / len(train_loader)
        train_loss_dict = dict(
            zip(_output_names + ['total'], avg_train_losses.tolist()))
        writer.add_scalars('train_loss', train_loss_dict, global_step=epoch)
        print('\nAverage training loss (epoch {}): {}'.format(
            epoch, train_loss_dict))

        val_losses = validate(model, validation_loader, device, criterion,
                              loss_size)
        avg_val_losses = val_losses / len(validation_loader)
        val_loss_dict = dict(
            zip(_output_names + ['total'], avg_val_losses.tolist()))
        writer.add_scalars('validation_loss', val_loss_dict, global_step=epoch)
        print('\nAverage validation loss (epoch {}): {}'.format(
            epoch, val_loss_dict))

        total_val_loss = val_losses[-1]
        lr_modifier.step(total_val_loss)

        if (epoch + 1) % save_every == 0:
            properties.update({'model_state_dict': model.state_dict()})
            properties.update({
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_dict,
                'val_loss': val_loss_dict,
                'epoch': epoch
            })
            torch.save(properties, save_file + ".e{}".format(epoch + 1))

    properties.update({'model_state_dict': model.state_dict()})
    properties.update({
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_dict,
        'val_loss': val_loss_dict,
        'epoch': epoch
    })
    torch.save(properties, save_file)


def _get_args():
    """Gets command line arguments"""
    project_path = os.path.abspath(os.path.join(deepab.__file__, "../.."))

    desc = ('''
        Script for training a model using a non-redundant set of bound and 
        unbound antibodies from SabDab with at most 99% sequence similarity, 
        a resolution cutoff of 3, and with a paired VH/VL.
        \n
        If there is no H5 file named antibody.h5 in the deepab/data directory, 
        then the script automatically uses the PDB files in 
        deepab/data/antibody_database directory to generate antibody.h5. If no 
        such directory exists, then the script downloads the set of pdbs from
        SabDab outlined above.
        ''')
    parser = argparse.ArgumentParser()
    # Model architecture arguments
    parser.add_argument('--num_blocks1D',
                        type=int,
                        default=3,
                        help='Number of one-dimensional ResNet blocks to use.')
    parser.add_argument('--num_blocks2D',
                        type=int,
                        default=25,
                        help='Number of two-dimensional ResNet blocks to use.')
    parser.add_argument('--dilation_cycle', type=int, default=5)
    parser.add_argument('--num_bins', type=int, default=37)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_gpu', default=False, action="store_true")
    parser.add_argument('--train_split', type=float, default=0.9)

    default_h5_file = os.path.join(project_path, 'data/abPwGeometry.h5')
    parser.add_argument('--h5_file', type=str, default=default_h5_file)
    default_antibody_database = os.path.join(project_path,
                                             'data/antibody_database')
    parser.add_argument('--antibody_database',
                        type=str,
                        default=default_antibody_database)

    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    default_model_path = os.path.join(project_path,
                                      'trained_models/model_{}/'.format(now))
    parser.add_argument('--output_dir', type=str, default=default_model_path)
    parser.add_argument('--pretrain_model_file', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=0)

    return parser.parse_args()


def _cli():
    args = _get_args()

    device_type = 'cuda' if torch.cuda.is_available(
    ) and args.use_gpu else 'cpu'
    device = torch.device(device_type)

    out_dir = args.output_dir
    current_epoch = 0
    if os.path.isdir(out_dir) and os.path.exists(
            os.path.join(out_dir, "model.p")):
        model_file = os.path.join(out_dir, "model.p")
        properties = torch.load(model_file, map_location='cpu')
        model = load_model(model_file, eval_mode=False,
                           device=device).to(device)

        current_epoch = properties['epoch'] + 1
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(properties['optimizer_state_dict'])
    elif args.pretrain_model_file != None and os.path.exists(
            args.pretrain_model_file):
        pretrain_model_file = args.pretrain_model_file
        properties = torch.load(pretrain_model_file, map_location='cpu')
        model = load_model(pretrain_model_file,
                           eval_mode=False,
                           device=device,
                           strict=False).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        properties.update({'lr': args.lr})

        print('Making {} ...'.format(out_dir))
        os.mkdir(out_dir)
    else:
        lstm_model_file = "trained_models/pairedseqlstm_scaler.p.e5"
        lstm_model = load_lstm_rnn(lstm_model_file, eval_mode=True).to(device)
        lstm_checkpoint_dict = torch.load(lstm_model_file, map_location='cpu')
        lstm_mean = torch.tensor(
            lstm_checkpoint_dict['scaler_mean']).float().to(device)
        lstm_scale = torch.tensor(
            lstm_checkpoint_dict['scaler_scale']).float().to(device)

        properties = dict(num_out_bins=args.num_bins,
                          num_blocks1D=args.num_blocks1D,
                          num_blocks2D=args.num_blocks2D,
                          dropout_proportion=args.dropout,
                          dilation_cycle=args.dilation_cycle)
        model = AbResNet(21,
                         lstm_model=lstm_model,
                         lstm_mean=lstm_mean,
                         lstm_scale=lstm_scale,
                         **properties)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        properties.update({'lr': args.lr})
        properties['lstm_checkpoint_dict'] = lstm_checkpoint_dict

        print('Making {} ...'.format(out_dir))
        os.mkdir(out_dir)

    properties.update({'lr': args.lr})

    # Load dataset loaders from h5 file
    h5_file = args.h5_file
    check_for_h5_file(h5_file, antibody_to_h5, args.antibody_database)
    dataset = H5PairwiseGeometryDataset(h5_file,
                                        num_bins=args.num_bins,
                                        mask_distant_orientations=True)
    train_split_length = int(len(dataset) * args.train_split)
    torch.manual_seed(args.random_seed)
    train_dataset, validation_dataset = data.random_split(
        dataset, [train_split_length,
                  len(dataset) - train_split_length])
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=H5PairwiseGeometryDataset.merge_samples_to_minibatch)
    validation_loader = data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        collate_fn=H5PairwiseGeometryDataset.merge_samples_to_minibatch)

    criterion = FocalLoss(ignore_index=dataset.mask_fill_value)

    lr_modifier = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             verbose=True)
    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))

    print('Arguments:\n', args)
    print('Model:\n', model)

    train(model=model,
          train_loader=train_loader,
          validation_loader=validation_loader,
          optimizer=optimizer,
          device=device,
          epochs=args.epochs,
          current_epoch=current_epoch,
          criterion=criterion,
          lr_modifier=lr_modifier,
          writer=writer,
          save_file=os.path.join(out_dir, 'model.p'),
          save_every=args.save_every,
          properties=properties)


if __name__ == '__main__':
    _cli()
