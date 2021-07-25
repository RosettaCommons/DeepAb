from typing import List, Tuple
import torch
from torch import nn


class ModelEnsemble(nn.Module):
    def __init__(self, load_model, model_files, eval_mode=False, device=None):
        super(ModelEnsemble, self).__init__()
        self.load_model = load_model
        self.model_files = model_files

        assert len(model_files) == 5

        if type(device) == type(None):
            for i, mf in enumerate(model_files):
                setattr(self, "model{}".format(i),
                        load_model(mf, eval_mode=eval_mode))
        else:
            for i, mf in enumerate(model_files):
                setattr(
                    self, "model{}".format(i),
                    load_model(mf, eval_mode=eval_mode,
                               device=device).to(device))

        try:
            self._num_out_bins = self.model0._num_out_bins
        except:
            self._num_out_bins = self.model0.num_out_bins

    def model_type(self):
        return type(self.model0)

    def forward(self, x):
        models = [
            self.model0, self.model1, self.model2, self.model3, self.model4
        ]
        out = [model.forward(x) for model in models]

        out_zip = [[o[i] for o in out] for i in range(len(out[0]))]
        out = [torch.mean(torch.stack(o), dim=0) for o in out_zip]

        return out