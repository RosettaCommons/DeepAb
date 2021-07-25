from typing import List
import torch
import torch.nn.functional as F


def max_shape(data: List[torch.Tensor]):
    """Gets the maximum length along all dimensions in a list of Tensors"""
    shapes = torch.stack([torch.tensor(_.shape) for _ in data])
    return torch.max(shapes.transpose(0, 1), dim=1)[0].int()


def pad_data_to_same_shape(tensor_list: List[torch.Tensor],
                           pad_value: int = 0):
    target_shape = max_shape(tensor_list)

    padded_dataset_shape = [len(tensor_list)]
    [padded_dataset_shape.append(_.item()) for _ in target_shape]
    padded_dataset = torch.empty(size=padded_dataset_shape).type_as(
        tensor_list[0])

    for i, data in enumerate(tensor_list):
        # Get how much padding is needed per dimension
        padding = torch.flip(target_shape - torch.tensor(data.shape), dims=[0])

        # Add 0 every other index to indicate only right padding
        padding = F.pad(padding.unsqueeze(0).t(), (1, 0, 0, 0)).view(-1, 1)
        padding: List[int] = padding.view(1, -1)[0].tolist()

        padded_data = F.pad(data, padding, value=float(pad_value))
        padded_dataset[i] = padded_data

    return padded_dataset
