import h5py
import math
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm

from deepab.build_fv.mds import get_full_dist_mat, fix_chirality
from deepab.util.tensor import pad_data_to_same_shape
from deepab.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins
from deepab.util.preprocess import bin_value_matrix
from deepab.util.masking import MASK_VALUE


class H5PairwiseGeometryDataset(data.Dataset):
    """
    Dataset containing sequence-structure pairs for training AbResNet
    """
    def __init__(self,
                 filename,
                 num_bins=37,
                 max_seq_len=None,
                 mask_distant_orientations=True,
                 mask_fill_value=MASK_VALUE):
        super(H5PairwiseGeometryDataset, self).__init__()

        self.filename = filename
        self.h5file = h5py.File(filename, 'r')
        self.num_proteins, _ = self.h5file['heavy_chain_primary'].shape

        self.num_bins = num_bins
        masked_bin_num = 1 if mask_distant_orientations else 0
        self.bins = [
            get_dist_bins(num_bins),
            get_dist_bins(num_bins),
            get_dist_bins(num_bins),
            get_dihedral_bins(num_bins - masked_bin_num),
            get_dihedral_bins(num_bins - masked_bin_num),
            get_planar_bins(num_bins - masked_bin_num)
        ]

        # Filter out sequences beyond the max length
        self.max_seq_len = max_seq_len
        self.valid_indices = None
        if max_seq_len is not None:
            self.valid_indices = self.get_valid_indices()
            self.num_proteins = len(self.valid_indices)

        self.mask_distant_orientations = mask_distant_orientations
        self.mask_fill_value = mask_fill_value

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise IndexError('Slicing not supported')

        if self.valid_indices is not None:
            index = self.valid_indices[index]

        id_ = self.h5file['id'][index]
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
        h3 = self.h5file['h3_range'][index]

        heavy_prim = F.one_hot(heavy_prim.long())
        light_prim = F.one_hot(light_prim.long())

        # Try to get the distance matrix from memory
        try:
            pairwise_geometry_mat = self.h5file['pairwise_geometry_mat'][
                index][:6, :total_seq_len, :total_seq_len]
            pairwise_geometry_mat = torch.Tensor(pairwise_geometry_mat).type(
                dtype=torch.float)
        except Exception:
            #euler_mat = generate_pnet_euler_matrix(tert, mask=mask)
            raise ValueError('Output matrix not defined')

        # Bin output matrices for classification or leave real values for regression
        nan_mat = torch.isnan(pairwise_geometry_mat)
        pairwise_geometry_mat = torch.stack([
            bin_value_matrix(mat, b)
            for mat, b in zip(pairwise_geometry_mat, self.bins)
        ])

        if self.mask_distant_orientations:
            distant_pairs = pairwise_geometry_mat[0] == self.num_bins - 1
            pairwise_geometry_mat[3:][:, distant_pairs] = self.num_bins - 1

        pairwise_geometry_mat[nan_mat] = self.mask_fill_value

        return id_, heavy_prim, light_prim, pairwise_geometry_mat, h3

    def get_valid_indices(self):
        """Gets indices with proteins less than the max sequence length
        :return: A list of indices with sequence lengths less than the max
                 sequence length.
        :rtype: list
        """
        valid_indices = []
        for i in range(self.h5file['heavy_chain_seq_len'].shape[0]):
            h_len = self.h5file['heavy_chain_seq_len'][i]
            l_len = self.h5file['light_chain_seq_len'][i]
            total_seq_len = h_len + l_len
            if total_seq_len < self.max_seq_len:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return self.num_proteins

    def get_class_weights(self):
        bin_counts = torch.zeros(6, self.num_bins).long()
        for _, _, _, pairwise_geometry_mat, _ in tqdm(self, total=len(self)):
            for i in range(pairwise_geometry_mat.shape[0]):
                bin_counts[i] += torch.bincount(pairwise_geometry_mat[i][
                    pairwise_geometry_mat[i] != self.mask_fill_value])

        class_weights = bin_counts.float() / bin_counts.sum(
            dim=1, keepdim=True).float()
        class_weights[class_weights == 0] = torch.mean(class_weights)

        return class_weights

    @staticmethod
    def merge_samples_to_minibatch(samples):
        # sort according to length of aa sequence
        samples.sort(key=lambda x: len(x[2]), reverse=True)
        return H5AntibodyBatch(zip(*samples)).data()


class H5AntibodyBatch:
    def __init__(self, batch_data, mask_fill_value=MASK_VALUE):
        (self.id_, self.heavy_prim, self.light_prim,
         self.pairwise_geometry_mat, self.h3) = batch_data
        self.mask_fill_value = mask_fill_value

    def data(self):
        return self.features(), self.labels()

    def features(self):
        """Gets the one-hot encoding of the sequences with a feature that
        delimits the chains"""
        X = [torch.cat(_, 0) for _ in zip(self.heavy_prim, self.light_prim)]
        X = pad_data_to_same_shape(X, pad_value=0).float()

        # Add chain delimiter
        X = F.pad(X, (0, 1, 0, 0, 0, 0))
        for i, h_prim in enumerate(self.heavy_prim):
            X[i, len(h_prim) - 1, X.shape[2] - 1] = 1

        # Switch shape from [batch, timestep/length, filter/channel]
        #                to [batch, filter/channel, timestep/length]
        return X.transpose(1, 2).contiguous()

    def labels(self):
        """Gets the distance matrix data of the batch with -1 padding"""
        label_mat = pad_data_to_same_shape(
            self.pairwise_geometry_mat,
            pad_value=self.mask_fill_value).transpose(0, 1)

        return label_mat

    def batch_mask(self):
        """Gets the mask data of the batch with zero padding"""
        '''Code to use when masks are added
        masks = self.mask
        masks = pad_data_to_same_shape(masks, pad_value=0)
        return masks
        '''
        raise NotImplementedError(
            'Masks have not been added to antibodies yet')


def h5_antibody_dataloader(filename,
                           batch_size=1,
                           max_seq_len=None,
                           num_bins=36,
                           **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError(
            'Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update(
        dict(collate_fn=H5PairwiseGeometryDataset.merge_samples_to_minibatch))
    kwargs.update(dict(batch_size=batch_size))

    return data.DataLoader(
        H5PairwiseGeometryDataset(filename,
                                  num_bins=num_bins,
                                  max_seq_len=max_seq_len), **kwargs)
