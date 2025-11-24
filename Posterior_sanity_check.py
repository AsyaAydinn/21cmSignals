import matplotlib.pyplot as plt
import corner
import numpy as np
import torch

def test_posterior(p, x_obs):
    print("  - type:", type(p))
    
    if p is None:
        print("!!! posterior = None ! Broken!")
        return False
      
    try:
        p.to("cpu")
        print(" posterior.to('cpu') OK")
    except Exception as e:
        print("!!! posterior.to('cpu') FAILED:", e)
        return False

    try:
        samp = p.sample((10,), x=x_obs.cpu())
        print("sample OK , shape:", samp.shape)
    except Exception as e:
        print("!!! sample FAILED:", e)
        return False

    print(" Posterior is okay")
    return True

i_obs = 8
x_obs = val_xs_global[i_obs:i_obs+1]

for s in seeds:
    test_posterior(posteriors[s], x_obs)

###########


device = torch.device("cpu")

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

embedding_net = ShellResNet().to("cpu")

seed = ##
posterior = posteriors[seed]

N_test = min(##, val_xs_global.shape[0])
num_samples_test = ###

val_xs_cpu     = val_xs_global[:N_test]              # (N_test, 2, H, W)
val_thetas_cpu = val_thetas_global[:N_test]          # (N_test, 3)


errors_mean = []  
inside_68 = []   
inside_95 = []

for i in range(N_test):
    x_val = val_xs_cpu[i:i+1].to(device)              # (1, 2, H, W)
    theta_true = val_thetas_cpu[i].numpy()            # (3,)

    with torch.no_grad():
        samples = posterior.sample(
            (num_samples_test,),
            x=x_val
        ).cpu().numpy()  # (num_samples_test, 3)

    theta_mean = samples.mean(axis=0)                 # (3,)
    errors_mean.append(theta_mean - theta_true)

    low68  = np.percentile(samples, 16, axis=0)
    high68 = np.percentile(samples, 84, axis=0)
    low95  = np.percentile(samples, 2.5, axis=0)
    high95 = np.percentile(samples, 97.5, axis=0)

    inside_68.append(
        np.logical_and(theta_true >= low68, theta_true <= high68)
    )
    inside_95.append(
        np.logical_and(theta_true >= low95, theta_true <= high95)
    )

errors_mean = np.stack(errors_mean, axis=0)            # (N_test, 3)
inside_68   = np.stack(inside_68,   axis=0)            # (N_test, 3) boolean
inside_95   = np.stack(inside_95,   axis=0)


param_names = [r"$\Omega_m$", r"$\Omega_b$", r"$\sigma_8$"]

for j in range(3):
    rmse = np.sqrt(np.mean(errors_mean[:, j]**2))
    cov68 = inside_68[:, j].mean() * 100.0
    cov95 = inside_95[:, j].mean() * 100.0

    print(f"Param {param_names[j]}:")
    print(f"  RMSE(posterior mean vs true) = {rmse:.4f}")
    print(f"  68% CI coverage ~ {cov68:.1f}% (ideal ≈ 68%)")
    print(f"  95% CI coverage ~ {cov95:.1f}% (ideal ≈ 95%)")
    print()

