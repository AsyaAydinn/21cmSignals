import os, re, json
import numpy as np
from pathlib import Path

POWER_ROOT = Path("###/power_spectrum")
OUTDIR     = Path("###/datasets/825_6145")
OUTDIR.mkdir(parents=True, exist_ok=True)
LH_PATH = Path("###/Quijote/latin_hypercube_params.txt")
P = np.loadtxt(LH_PATH, comments="#")

ELL_BAND = (82.5, 614.5)
SHELL_RMIN, SHELL_RMAX = 240, 990
FLOOR = 1e-12   
SEED = 42

NPZ_NAME_RE = re.compile(r"Tb_shell(\d+)_(\d+)_dTB_cl\.npz$")

def find_shell_files(sim_dir: Path):
    shells = []
    for fname in os.listdir(sim_dir):
        m = NPZ_NAME_RE.match(fname)
        if not m:
            continue
        shell_in, shell_out = map(int, m.groups())
        if SHELL_RMIN <= shell_in < SHELL_RMAX:
            shells.append((shell_in, shell_out, sim_dir / fname))
    return sorted(shells)


FITS_ROOT = Path("###/fits")

def read_shell_mean_tb(sim_id: int, shell_in: int, shell_out: int) -> float:
    meta_path = FITS_ROOT / f"{sim_id}" / f"Tb_shell{shell_in}_{shell_out}_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return float(meta["mean_Tb"])


def load_Dl_matrix(sim_dir: Path, ell_ref: np.ndarray):
    sim_id = int(sim_dir.name)
    shells = find_shell_files(sim_dir)
    X_rows = []
    tb_means = []                                          # per-shell mean Tb (mK)

    for shell_in, shell_out, fpath in shells:
        d = np.load(fpath)
        ell, Dl = d["ell"], d["Dl"]
        m = (ell >= ELL_BAND[0]) & (ell <= ELL_BAND[1])
        ell, Dl = ell[m], Dl[m]
        if ell_ref.size == 0:
            ell_ref = ell.copy()
        else:
            if not np.allclose(ell, ell_ref, rtol=1e-3, atol=1e-3):
                raise ValueError(f"Inconsistent ell grid in {fpath}")

        Dl = np.clip(Dl, FLOOR, None)
        X_rows.append(np.log10(Dl))

        tb_mean_mk = read_shell_mean_tb(sim_id, shell_in, shell_out)
        tb_means.append(tb_mean_mk)

    X = np.array(X_rows, dtype=np.float32)                 # (n_shells, n_bins)
    tb_means = np.array(tb_means, dtype=np.float32)        # (n_shells,)
    return X, tb_means, ell_ref



sim_dirs = [POWER_ROOT / f for f in os.listdir(POWER_ROOT) if f.isdigit()]
ell_ref = np.array([])
all_X, all_tb, all_ids = [], [], []
for sim_dir in sorted(sim_dirs, key=lambda p: int(p.name)):
    try:
        X, tb_means, ell_ref = load_Dl_matrix(sim_dir, ell_ref)
        all_X.append(X)
        all_tb.append(tb_means)
        all_ids.append(int(sim_dir.name))
    except Exception as e:
        print(f"Skipping {sim_dir}: {e}")

all_X  = np.array(all_X)                                    # (N, n_shells, n_bins)
all_tb = np.array(all_tb)                                   # (N, n_shells)
print("Dataset shapes:", all_X.shape, all_tb.shape)


rng = np.random.default_rng(SEED)
perm = rng.permutation(len(all_X))
train_frac = 1
n_train = int(train_frac * len(all_X))
train_idx, val_idx = perm[:n_train], perm[n_train:]

train_data_X  = all_X[train_idx]                            # (Ntr, S, L)
train_data_tb = all_tb[train_idx]                           # (Ntr, S)


mu_X    = train_data_X.mean(axis=(0, 2), keepdims=True)      # (S,1)
sigma_X = train_data_X.std(axis=(0, 2), keepdims=True)
sigma_X[sigma_X < 1e-8] = 1e-8
log_tb  = np.log10(train_data_tb)
mu_tb   = log_tb.mean(axis=0)                                # (S,)
sigma_tb = log_tb.std(axis=0)
sigma_tb[sigma_tb < 1e-8] = 1e-8

def norm_X(X):   return (X - mu_X) / sigma_X                 # (S, L)
def norm_tb(tb): return (np.log10(tb) - mu_tb) / sigma_tb    # (S,)


for X, tb, sid in zip(all_X, all_tb, all_ids):
    Xn  = norm_X(X)
    tbn = norm_tb(tb)
    tb_chan = np.repeat(tbn[:, None], X.shape[1], axis=1).astype(np.float32)

    np.savez_compressed(
        OUTDIR / f"sim_{sid:04d}.npz",
        Dl_norm=Xn.astype(np.float32),
        Tbch_norm=tb_chan
    )


meta = dict(
    ell=ell_ref.tolist(),
    mu_X=mu_X.squeeze().tolist(),
    sigma_X=sigma_X.squeeze().tolist(),
    mu_tb=mu_tb.tolist(),
    sigma_tb=sigma_tb.tolist(),
    ell_band=ELL_BAND,
    shell_range=(SHELL_RMIN, SHELL_RMAX),
    train_ids=[int(x) for x in np.array(all_ids)[train_idx]],
    val_ids=[int(x) for x in np.array(all_ids)[val_idx]],
    note="Dl_norm normalized per-shell (μ,σ computed across all ℓ), "
         "Tbch_norm is per-shell log10(meanTb) normalized and broadcast across ℓ."
)
with open(OUTDIR / "preproc_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f" dataset saved to {OUTDIR}")



#################


files = sorted([f for f in os.listdir(OUTDIR) if f.endswith(".npz")])
X_list, ids = [], []

for f in files:
    sid = int(f.split("_")[1].split(".")[0])
    d = np.load(OUTDIR / f)
    Dl = np.squeeze(d["Dl_norm"])    
    Tb = np.squeeze(d["Tbch_norm"]) 
    X_list.append(np.stack([Dl, Tb], axis=0)) 
    ids.append(sid)

X = np.stack(X_list, axis=0) 
ids = np.array(ids)
Y = P[ids][:, [0, 1, 4]]  # Omega_m, Omega_b, sigma_8

print("Final shapes:", X.shape, Y.shape)
np.save(OUTDIR / "X.npy", X)
np.save(OUTDIR / "Y.npy", Y)

