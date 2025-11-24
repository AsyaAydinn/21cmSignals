import matplotlib.pyplot as plt
import corner
import numpy as np
import torch
from scipy.stats import binom

samples = ensemble_samples.detach().cpu().numpy()
# samples = torch.load(f"{OUT_DIR}/ensemble_samples.pt").cpu().numpy()

theta_true = Y[8].detach().cpu().numpy()
labels = [r"$\Omega_m$", r"$\Omega_b$", r"$\sigma_8$"]
fig = corner.corner(
    samples,
    labels=labels,
    truths=theta_true,
    color="royalblue",
    truth_color="crimson",
    show_titles=True,
    title_fmt=".3f",
    smooth=1.0
)

plt.show()




################




def ensemble_sbc(posteriors, seeds, thetas, xs, n_samples=50000, max_n_val=None):
    thetas = thetas.cpu()
    xs     = xs.cpu()

    N_val, D = thetas.shape
    if max_n_val is not None:
        N_use = min(max_n_val, N_val)
    else:
        N_use = N_val

    S = len(seeds)
    N_tot = n_samples * S  

    ranks = np.zeros((N_use, D), dtype=np.int64)

    for i in range(N_use):
        theta_true = thetas[i]      
        x_i        = xs[i:i+1]          

        all_samp = []
        for s in seeds:
            p = posteriors[s]
            samp = p.sample((n_samples,), x=x_i).cpu()   
            all_samp.append(samp)

        all_samp = torch.cat(all_samp, dim=0)         
        for d in range(D):
            ranks[i, d] = (all_samp[:, d] <= theta_true[d]).sum().item()

    return ranks, N_tot


def sbc_hist_with_band(u, n_bins=15, label=r"$\theta$", alpha_band=0.99):
    N = u.shape[0]              
    counts, edges = np.histogram(u, bins=n_bins, range=(0, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])

    p = 1.0 / n_bins
    q_low  = (1 - alpha_band) / 2          # 0.005
    q_high = 1 - q_low                     # 0.995

    lower = binom.ppf(q_low,  N, p) / (N * (1/n_bins))
    upper = binom.ppf(q_high, N, p) / (N * (1/n_bins))


    bin_width = 1.0 / n_bins
    counts_density = counts / (N * bin_width)
    plt.bar(centers, counts_density, width=bin_width, alpha=0.6, label="SBC")
    expected_density = (N*p) / (N * bin_width)
    plt.axhline(expected_density, linestyle="--", label="ideal")
    plt.fill_between(
        centers,
        lower,
        upper,
        color="gray",
        alpha=0.25,
        label="Binomial 99% band"
    )

    plt.ylim(bottom=0)
    plt.xlabel("rank / (N+1)")
    plt.ylabel("density")
    plt.title(f"SBC – {label}")
    plt.legend()

ranks, N_tot = ensemble_sbc(
    posteriors=posteriors,
    seeds=seeds,
    thetas=val_thetas_global,
    xs=val_xs_global,
    n_samples=####,
)

u = (ranks + 0.5) / (N_tot + 1.0)                                   # shape [N_use, 3]

param_labels = [r"$\Omega_m$", r"$\Omega_b$", r"$\sigma_8$"]
for d in range(u.shape[1]):
    plt.figure()
    sbc_hist_with_band(u[:, d], n_bins=15, label=param_labels[d])
    plt.show()

