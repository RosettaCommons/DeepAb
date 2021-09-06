import h5py
import torch
import torch.utils.data as data
import torch.nn.functional as F

from deepab.util.tensor import pad_data_to_same_shape


class H5PairedSeqDataset(data.Dataset):
    """
    Dataset containing paired sequence for training PairedSeqLSTM
    """
    def __init__(self, filename):
        super(H5PairedSeqDataset, self).__init__()

        self.filename = filename
        self.h5file = h5py.File(filename, 'r')
        self.num_proteins, _ = self.h5file['heavy_chain_primary'].shape

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise IndexError('Slicing not supported')

        heavy_seq_len = self.h5file['heavy_chain_seq_len'][index]
        light_seq_len = self.h5file['light_chain_seq_len'][index]
        total_seq_len = heavy_seq_len + light_seq_len

        # Get the attributes from a protein and cut off zero padding
        heavy_prim = self.h5file['heavy_chain_primary'][index, :heavy_seq_len]
        light_prim = self.h5file['light_chain_primary'][index, :light_seq_len]

        # Convert to torch tensors
        heavy_prim = torch.Tensor(heavy_prim).type(dtype=torch.uint8)
        light_prim = torch.Tensor(light_prim).type(dtype=torch.uint8)

        # Get CDR loops
        h1 = self.h5file['h1_range'][index]
        h2 = self.h5file['h2_range'][index]
        h3 = self.h5file['h3_range'][index]
        l1 = self.h5file['h1_range'][index]
        l2 = self.h5file['h2_range'][index]
        l3 = self.h5file['h3_range'][index]
        cdrs = [h1, h2, h3, l1, l2, l3]

        cdr_mask = torch.zeros(total_seq_len)
        for cdr in cdrs:
            cdr_mask[cdr[0]:cdr[1] + 1] = 1

        metadata = {
            'species': str(self.h5file['species'][index], 'utf-8'),
            'isotype': str(self.h5file['isotype'][index], 'utf-8'),
            'b_type': str(self.h5file['b_type'][index], 'utf-8'),
            'b_source': str(self.h5file['b_source'][index], 'utf-8'),
            'disease': str(self.h5file['disease'][index], 'utf-8'),
            'vaccine': str(self.h5file['vaccine'][index], 'utf-8')
        }

        return index, heavy_prim, light_prim, cdrs, metadata

    def __len__(self):
        return self.num_proteins

    @staticmethod
    def merge_samples_to_minibatch(samples):
        return H5AntibodyBatch(zip(*samples)).data()


class H5AntibodyBatch:
    def __init__(self, batch_data):
        (self.index, self.heavy_prim, self.light_prim, self.cdrs,
         self.metadata) = batch_data

    def data(self):
        return self.features(), self.labels(), self.index

    def features(self):
        seq_start, seq_end, seq_delim = torch.tensor(
            [20]).byte(), torch.tensor([21]).byte(), torch.tensor([22]).byte()

        combined_seqs = [
            torch.cat([seq_start, h, seq_delim, l, seq_end])
            for h, l in zip(self.heavy_prim, self.light_prim)
        ]
        combined_seqs = pad_data_to_same_shape(combined_seqs, pad_value=22)
        combined_seqs = torch.stack(
            [F.one_hot(seq.long()) for seq in combined_seqs])

        combined_seqs = combined_seqs.transpose(0, 1)

        return combined_seqs

    def labels(self):
        seq_start, seq_end, seq_delim = torch.tensor(
            [20]).byte(), torch.tensor([21]).byte(), torch.tensor([22]).byte()

        combined_seqs = [
            torch.cat([seq_start, h, seq_delim, l, seq_end])
            for h, l in zip(self.heavy_prim, self.light_prim)
        ]
        combined_seqs = pad_data_to_same_shape(combined_seqs, pad_value=22)

        return combined_seqs.long()


def h5_antibody_dataloader(filename, batch_size=1, **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError(
            'Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update(
        dict(collate_fn=H5PairedSeqDataset.merge_samples_to_minibatch))
    kwargs.update(dict(batch_size=batch_size))

    return data.DataLoader(H5PairedSeqDataset(filename), **kwargs)
