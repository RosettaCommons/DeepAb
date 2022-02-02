import argparse
import torch
import os
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import deepab
from deepab.models.PairedSeqLSTM import PairedSeqLSTM
from deepab.util.util import RawTextArgumentDefaultsHelpFormatter
from deepab.datasets.H5PairedSeqDataset import H5PairedSeqDataset


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Trains a model for one epoch"""
    model.train()
    running_loss = 0
    e_i = 0
    for inputs, labels, _ in tqdm(train_loader, total=len(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)[:, 1:]

        optimizer.zero_grad()

        def handle_batch():
            """Function done to ensure variables immediately get dealloced"""
            output = model(src=inputs, trg=inputs)
            output = output[1:].permute(1, 2, 0)

            loss = criterion(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            return loss.item()

        loss = handle_batch()
        running_loss += loss

        if e_i % 100 == 0:
            print(loss)
        e_i += 1
        # running_loss += handle_batch()

    return running_loss


def validate(model, validation_loader, criterion, device):
    """"""
    with torch.no_grad():
        model.eval()
        running_loss = 0
    for inputs, labels, _ in tqdm(validation_loader,
                                  total=len(validation_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)[:, 1:]

        def handle_batch():
            """Function done to ensure variables immediately get dealloced"""
            output = model(src=inputs, trg=inputs)
            output = output[1:].permute(1, 2, 0)

            loss = criterion(output, labels)

            return loss.item()

        running_loss += handle_batch()

    return running_loss


def train(model,
          train_loader,
          validation_loader,
          criterion,
          optimizer,
          epochs,
          device,
          lr_modifier,
          writer,
          save_file,
          save_every,
          properties=None):
    """"""
    properties = {} if properties is None else properties
    print('Using {} as device'.format(str(device).upper()))
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 device)
        avg_train_loss = train_loss / len(train_loader)
        train_loss_dict = {"cce": avg_train_loss}
        writer.add_scalars('train_loss', train_loss_dict, global_step=epoch)
        print('\nAverage training loss (epoch {}): {}'.format(
            epoch, avg_train_loss))

        val_loss = validate(model, validation_loader, criterion, device)
        avg_val_loss = val_loss / len(validation_loader)
        val_loss_dict = {"cce": avg_val_loss}
        writer.add_scalars('validation_loss', val_loss_dict, global_step=epoch)
        print('\nAverage validation loss (epoch {}): {}'.format(
            epoch, avg_val_loss))

        lr_modifier.step(val_loss)

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


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def _get_args():
    """Gets command line arguments"""
    project_path = os.path.abspath(os.path.join(deepab.__file__, "../.."))

    desc = ('''
        Desc pending
        ''')
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=RawTextArgumentDefaultsHelpFormatter)
    # Model architecture arguments
    parser.add_argument('--enc_hid_dim', type=int, default=64)
    parser.add_argument('--dec_hid_dim', type=int, default=64)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_gpu', default=False, action="store_true")
    parser.add_argument('--train_split', type=float, default=0.95)

    default_h5_file = os.path.join(project_path, 'data/abSeq.h5')
    parser.add_argument('--h5_file', type=str, default=default_h5_file)

    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    default_model_path = os.path.join(project_path,
                                      'trained_models/model_{}/'.format(now))
    parser.add_argument('--output_dir', type=str, default=default_model_path)
    return parser.parse_args()


def _cli():
    """Command line interface for train.py when it is run as a script"""
    args = _get_args()
    device_type = 'cuda' if torch.cuda.is_available(
    ) and args.use_gpu else 'cpu'
    device = torch.device(device_type)

    properties = dict(seq_dim=23,
                      enc_hid_dim=args.enc_hid_dim,
                      dec_hid_dim=args.dec_hid_dim)

    model = PairedSeqLSTM(**properties)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    properties.update({'lr': args.lr})

    # Load dataset loaders from h5 file
    h5_file = args.h5_file
    dataset = H5PairedSeqDataset(h5_file)
    train_split_length = int(len(dataset) * args.train_split)
    torch.manual_seed(0)
    train_dataset, validation_dataset = data.random_split(
        dataset, [train_split_length,
                  len(dataset) - train_split_length])
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=H5PairedSeqDataset.merge_samples_to_minibatch)
    validation_loader = data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        collate_fn=H5PairedSeqDataset.merge_samples_to_minibatch)

    lr_modifier = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             verbose=True)
    out_dir = args.output_dir
    if not os.path.isdir(out_dir):
        print('Making {} ...'.format(out_dir))
        os.mkdir(out_dir)
    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))

    print('Arguments:\n', args)
    print('Model:\n', model)

    train(model=model,
          train_loader=train_loader,
          validation_loader=validation_loader,
          criterion=criterion,
          optimizer=optimizer,
          device=device,
          epochs=args.epochs,
          lr_modifier=lr_modifier,
          writer=writer,
          save_file=os.path.join(out_dir, 'model.p'),
          save_every=args.save_every,
          properties=properties)


if __name__ == '__main__':
    _cli()
