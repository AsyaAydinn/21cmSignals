import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, os

from torch.utils.data import DataLoader, TensorDataset, random_split
from sbi import utils as U
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn



OUT_DIR = "/project/ag-weller/Asya.Aydin/snpe_outputs/825_6145"
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = np.load("###/datasets/825_6145/X.npy")
Y = np.load("###/datasets/825_6145/Y.npy")


X = torch.tensor(X, dtype=torch.float32, device=device)                                  # GPU
Y = torch.tensor(Y, dtype=torch.float32, device=device)  
prior = U.BoxUniform(
    low=torch.tensor([0.1, 0.03, 0.6], dtype=torch.float32, device=device),
    high=torch.tensor([0.5, 0.07, 1.0], dtype=torch.float32, device=device)
)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
        )
        self.act = nn.LeakyReLU()
    def forward(self, x):
        return self.act(x + self.conv(x))


class ShellResNet(nn.Module):
    def __init__(self, emb_dim=60):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.block1 = ResBlock(64)
        self.block2 = ResBlock(64)
        self.block3 = ResBlock(64)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 4 * 4, emb_dim)
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)



class SNPE_with_dataloader(SNPE):
    def get_dataloaders(self, *args, **kwargs):
        loaders = super().get_dataloaders(*args, **kwargs)
        if isinstance(loaders, tuple) and len(loaders) == 2:
            self.train_loader, self.val_loader = loaders
        return loaders



def run_snpe(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    embedding_net = ShellResNet().to(device)
    print(device)
    print(next(embedding_net.parameters()).device)

    density_estimator_build = posterior_nn(
        model="nsf",
        embedding_net=embedding_net
    )

    inference = SNPE_with_dataloader(
        prior=prior,
        density_estimator=density_estimator_build,
        device=device
    )

    noise = 0.0
    X_noisy = X + noise
    inference.append_simulations(Y, X_noisy)

    density_estimator = inference.train(
        max_num_epochs=200,
        training_batch_size=50,
        validation_fraction=0.15,
    )


    train_loader = inference.train_loader
    val_loader   = inference.val_loader  

    if val_loader is not None:
        xs_list, theta_list = [], []
        
        for batch in val_loader:
            if len(batch) == 2:
                theta_batch, x_batch = batch
            elif len(batch) == 3:
                theta_batch, x_batch, _ = batch                                    #ignore weight
            else:
                raise RuntimeError(f"batch size: {len(batch)}")
        
            xs_list.append(x_batch.cpu())
            theta_list.append(theta_batch.cpu())

        val_xs = torch.cat(xs_list, dim=0)
        val_thetas = torch.cat(theta_list, dim=0)
    else:
        val_xs, val_thetas = None, None

    posterior = inference.build_posterior(density_estimator)
    return posterior, val_xs, val_thetas



seeds = [2]
posteriors = {}
val_xs_global, val_thetas_global = None, None

for i, s in enumerate(seeds):
    print(f"Running seed = {s}", flush=True)
    posterior, val_xs, val_thetas = run_snpe(s)

    posteriors[s] = posterior  

    if i == 0:

        val_xs_global = val_xs
        val_thetas_global = val_thetas

torch.save(posteriors, f"{OUT_DIR}/posteriors.pt")


x_obs = (X + 0.0)[8:9]
num_samples = 1000000  

all_samp = []
for s in seeds:
    print(f"Sampling from seed {s}", flush=True)
    p = posteriors[s]
    samp = p.sample((num_samples,), x=x_obs).cpu()
    all_samp.append(samp)

ensemble_samples = torch.cat(all_samp, dim=0)
torch.save(ensemble_samples, f"{OUT_DIR}/ensemble_samples.pt")
