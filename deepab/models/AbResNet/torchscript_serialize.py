import torch

from deepab.models.AbResNet import load_model
from deepab.models.ModelEnsemble5 import ModelEnsemble

# model = load_model("trained_models/ensemble_abresnet/rs0.pt", eval_mode=True)
model = ModelEnsemble(model_files=[
    "trained_models/ensemble_abresnet/rs0.pt",
    "trained_models/ensemble_abresnet/rs1.pt",
    "trained_models/ensemble_abresnet/rs2.pt",
    "trained_models/ensemble_abresnet/rs3.pt",
    "trained_models/ensemble_abresnet/rs4.pt"
],
                      load_model=load_model,
                      eval_mode=True)

x = torch.randn((1, 21, 230))
sm = torch.jit.script(model, x)
sm.save("ts_ensemble_abresnet.pt")