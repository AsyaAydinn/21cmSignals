#Generates 21 cm temperature and temperature fluctuation maps for 2000 N body Simulations at redshift 1

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import h as h_planck, c, k, physical_constants
from astropy.io import fits
import json

NSIDE = 512

origin = np.array([0.0, 0.0, 0.0])
base_input = "###"
latin_hypercube_file = "###"
output_base = "###"
z = 1

shell_start = 0.0
shell_end = 1000.0
shell_thickness = 30.0      
shell_edges = np.arange(shell_start, shell_end, shell_thickness)


Y_p = 0.24
alpha = 0.2
beta = -0.58
v_c0 = 10**1.56
Msun_to_kg = 1.988e30
Mpc_to_m = 3.086e22
A10 = 2.869e-15
f21 = 1.4204065e9
m_p = physical_constants['proton mass'][0]
k_b = k

os.makedirs(os.path.join(output_base, "brightness_temp"), exist_ok=True)
os.makedirs(os.path.join(output_base, "fits"), exist_ok=True)


def v_circ(M, z, Omega_m, h):
    factor = (200 * Omega_m * h**2 / 24.4)**(1/6)
    return 96.6 * factor * ((1+z)/3.3)**0.5 * (M / 1e11)**(1/3)


def M_HI(M, z, Omega_m, f_H_c, h):
    mass_ratio = (M / (1e11/h))**beta   # Msun/h 
    vc = v_circ(M / h, z, Omega_m, h)   # Msuna 
    return alpha * f_H_c * M * mass_ratio * np.exp(-(v_c0 / vc)**3)

def load_latin_hypercube_params(filepath):
    try:
        return np.loadtxt(filepath, comments="#") 
    except Exception as e:
        print(f"Error loading Latin hypercube params: {e}")
        return None


latin_data = np.loadtxt(latin_hypercube_file, comments="#", dtype=np.float64)

all_folders = sorted([
    f for f in os.listdir(base_input)
    if f.isdigit() and 0 <= int(f) <= 2000
], key=int)


for folder in all_folders:
    halo_file = os.path.join(base_input, folder, "halos.txt")
    if not os.path.exists(halo_file):
        continue
    print(f"\nProcessing sim {folder}")

    try:
        folder_num = int(folder)
        param_index = folder_num


        if param_index >= latin_data.shape[0]:
            print(f"Skipping {folder}: param index {param_index} out of bounds.")
            continue

        Omega_m, Omega_b, h, n_s, sigma_8 = latin_data[param_index]
        Omega_L = 1 - Omega_m
        f_H_c = (1 - Y_p) * Omega_b / Omega_m
        H0 = 100 * h *1e3 / Mpc_to_m    #m /s/m
        Hz = H0  * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)   #
        Tb_pre = (3 * h_planck * c**3 * A10) / (32  *np.pi * k_b * f21**2 * m_p)  # K·m^3·kg^-1·s^-1

        data = np.loadtxt(halo_file, skiprows=6)
        pos = data[:, 1:4] #comov
        mass = data[:, 7] #msun/h
        Lbox = 1000.0  # Mpc/h
        pos_corner = pos    
        r_all= np.linalg.norm(pos_corner, axis=1)
        

        Npix = hp.nside2npix(NSIDE)
        theta_p, phi_p = hp.pix2ang(NSIDE, np.arange(Npix))
        octant_mask = (theta_p <= np.pi/2) & (phi_p <= np.pi/2)
        Npix_octant = int(octant_mask.sum())
        f_sky        = float(Npix_octant / Npix)

        for shell_in in shell_edges:
            shell_out = min(shell_in + shell_thickness, shell_end)
            shell_mask = (r_all >= shell_in) & (r_all < shell_out)
            if np.sum(shell_mask) == 0:
                continue

            pos_shell_corner = pos_corner[shell_mask]
            mass_shell = mass[shell_mask]
            r_shell = np.linalg.norm(pos_shell_corner, axis =1 )

            x, y, zc = pos_shell_corner.T
            cosang = np.clip(zc / r_shell, -1.0, 1.0)  
            theta = np.arccos(cosang)
            phi = np.arctan2(y, x)
            phi = np.where(phi < 0, phi + 2 * np.pi, phi)
           
            pix = hp.ang2pix(NSIDE, theta, phi)


            
            M_HI_vals = M_HI(mass_shell, z, Omega_m, f_H_c,h)      # msun/h 
        
            mass_map = np.bincount(pix, weights=M_HI_vals, minlength=Npix).astype(np.float64)  #  Msun/h
            mass_map[~octant_mask] = 0.0
            r_center = 0.5 * (shell_in + shell_out)  #
            dr = (shell_out - shell_in)


            shell_volume = (4*np.pi * r_center**2 * dr) * f_sky / (1 + z)**3   #proper volume
            vol_pix = shell_volume / Npix_octant       
            vol_pix_m3 = vol_pix * (Mpc_to_m**3) / (h**3)    

            mass_map_kg = mass_map* Msun_to_kg / h                            # [kg] 
            rho_HI = mass_map_kg / vol_pix_m3                                 # [kg/m^3] comoving
            Tb = Tb_pre * ((1 + z)**2 / Hz) * rho_HI* 1e3                     # milikelvin 
            
            
            mean_Tb = Tb[octant_mask].mean()
            delta_Tb = np.zeros_like(Tb, dtype=np.float32)      
            delta_Tb[octant_mask] = (Tb[octant_mask] - mean_Tb) / mean_Tb
            Tb_map = Tb.astype(np.float32)
            delta_map = delta_Tb.astype(np.float32)
            Tb_map[octant_mask] = Tb[octant_mask].astype(np.float32)
            delta_map[octant_mask] = delta_Tb[octant_mask].astype(np.float32)



            M_HI_tot_Msun_h = float(M_HI_vals.sum())         
            M_HI_tot_kg     = M_HI_tot_Msun_h * Msun_to_kg / h # kg
            shell_volume_m3 = shell_volume * (Mpc_to_m**3) / (h**3)  
            rho_HI_mean = float(M_HI_tot_kg / shell_volume_m3)       

            N_halos_shell   = int(len(mass_shell))


            plot_dir = os.path.join(output_base, "brightness_temp", folder)
            fits_dir = os.path.join(output_base, "fits", folder)
            os.makedirs(plot_dir, exist_ok=True)
            os.makedirs(fits_dir, exist_ok=True)

            plot_name = f"Tb_shell{int(shell_in)}_{int(shell_out)}.png"
            fits_name = f"Tb_shell{int(shell_in)}_{int(shell_out)}.fits"



            Tb_plot = Tb_map.copy()
            valid = (Tb_plot != hp.UNSEEN)
            p1, p99 = np.percentile(Tb_plot[valid], [1, 99]) 

            Tb_plot[valid] = np.clip(Tb_plot[valid], 5e-4, None)
            fwhm = np.radians(0.5)  # 0.5° beam
            Tb_plot = hp.smoothing(Tb_plot, fwhm=fwhm, verbose=False)
            title = f"T_b log (shell {shell_in:.0f}-{shell_out:.0f}) | Omega_m={Omega_m:.4f}, Omega_b={Omega_b:.4f}, h={h:.4f}, z={z:.1f}"
            hp.mollview(Tb_plot + 1e-6, cbar=False, title=title,
                        unit='mK', cmap='inferno', min=p1, max=p99)

            fig = plt.gcf()
            ax = plt.gca()
            image = ax.get_images()[0]
            image.set_norm(LogNorm(vmin=1e-4, vmax=1))
            cbar = fig.colorbar(image, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)
            cbar.set_label("Brightness Temperature [mK] (log scale)", fontsize=12, labelpad=8)
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.xaxis.set_ticks_position('bottom')
            plt.savefig(os.path.join(plot_dir, plot_name))
            plt.clf()
            plt.close()


            shell_name = f"Tb_shell{int(shell_in)}_{int(shell_out)}"
            fits_path = os.path.join(fits_dir, f"{shell_name}.fits")
            hp.write_map(fits_path, [Tb_map.astype(np.float32), delta_map.astype(np.float32)],
            column_names=['TB_mK','DTB'],
            dtype=np.float32,
            overwrite=True
        )
           
            metadata = dict(
                shell_in=float(shell_in),
                shell_out=float(shell_out),
                mass_mean=float(mass_shell.mean()),
                mass_var=float(mass_shell.var()),
                mass_mean_M_HI=float(M_HI_vals.mean()),
                mass_var_M_HI=float(M_HI_vals.var()),
                M_HI_tot_Msun_h=M_HI_tot_Msun_h,

                N_halos=int(N_halos_shell),
                f_sky = f_sky,
                
                mean_rho_HI_kg_m3=rho_HI_mean,
                mean_Tb=float(mean_Tb),
                model_params=dict(alpha=float(alpha), beta=float(beta), v_c0=float(v_c0), Y_p=float(Y_p)),
                cosmo_params=dict(Omega_m=float(Omega_m), Omega_b=float(Omega_b),
                                  h=float(h), n_s=float(n_s), sigma_8=float(sigma_8), z=float(z))
         
            )


            meta_path = os.path.join(fits_dir, f"{shell_name}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)

            mask_path = os.path.join(fits_dir, f"{shell_name}_mask.fits")
            hp.write_map(mask_path, octant_mask.astype(np.float32), overwrite=True)
            
    except Exception as e:
        print(f"Error in sim {folder}: {e}")
