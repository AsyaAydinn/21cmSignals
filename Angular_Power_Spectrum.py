import os, re, hashlib
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt  

NSIDE      = 512
LMAX       = 3*NSIDE - 1
BIN_WIDTH  = 6
INPUT_ROOT = "###"
OUTPUT_ROOT= "###"
SIM_MIN, SIM_MAX = 501, 750  
import json
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def mask_key(mk: np.ndarray) -> str:
    return hashlib.sha1(mk.view(np.uint8)).hexdigest()


shell_re = re.compile(r"Tb_shell(\d+)_(\d+)\.fits$")
binning = nmt.NmtBin.from_lmax_linear(lmax=LMAX, nlb=BIN_WIDTH)
ell_b   = binning.get_effective_ells()

workspace_cache = {} 
saved_ell_once  = False

def fsky_eff(mask: np.ndarray) -> float:
    return float(np.mean(mask**2))


#Knox formula for bandpower variance on a masked sky 
def knox_sigma(ell_b: np.ndarray, Cb: np.ndarray, bin_width: int, fsky: float) -> np.ndarray:
    lb = np.maximum(ell_b, 1.0) 
    var = 2.0 / ((2.0*lb + 1.0) * bin_width * max(fsky, 1e-6)) * (Cb**2)
    return np.sqrt(var)


for folder in sorted(os.listdir(INPUT_ROOT), key=lambda x: int(x) if x.isdigit() else 10**9):
    if not folder.isdigit():
        continue
    fid = int(folder)
    if not (SIM_MIN <= fid <= SIM_MAX):
        continue

    folder_path = os.path.join(INPUT_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue
    fits_files = []
    for f in os.listdir(folder_path):
        if not f.endswith(".fits") or "_mask" in f:
            continue
        m = shell_re.match(f)
        if not m:
            continue
        shell_in  = int(m.group(1))
        shell_out = int(m.group(2))
        if (shell_in >= 240) and (shell_out <= 990):
            fits_files.append((shell_in, shell_out, f))
    fits_files.sort()
    if not fits_files:
        continue
    

    out_dir = os.path.join(OUTPUT_ROOT, f"{fid}")
    os.makedirs(out_dir, exist_ok=True)


    for shell_in, shell_out, fname in fits_files:
        fpath = os.path.join(folder_path, fname)
        shell_name = fname[:-5]
        TB_mK, DTB = hp.read_map(fpath, field=(0, 1), dtype=np.float64, verbose=False)
        
        meta_path = fpath.replace(".fits", "_meta.json")
        with open(meta_path, "r") as _f:
            meta = json.load(_f)
        mean_Tb_mK = float(meta["mean_Tb"]) 
        

        mask_path = fpath.replace(".fits", "_mask.fits")
        if not os.path.exists(mask_path):
            mk = np.isfinite(TB_mK) & (TB_mK != hp.UNSEEN)
            mask = mk.astype(np.float64)
        else:
            mask_raw = hp.read_map(mask_path, dtype=np.float64, verbose=False)
            mask = np.where((mask_raw == hp.UNSEEN) | (mask_raw <= 0.0), 0.0, 1.0)


        mask = nmt.mask_apodization(mask, aposize=1.0, apotype="C2")


        if (mask > 0).sum() < 10:
            continue

        mkey = mask_key(mask.astype(np.float32))
        if mkey in workspace_cache:
            w = workspace_cache[mkey]
        else:
            f_tmp = nmt.NmtField(mask, [mask*0.0])  
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(f_tmp, f_tmp, binning)
            workspace_cache[mkey] = w


        dtb = DTB.copy()
        dtb = np.where(np.isfinite(dtb), dtb, 0.0)
        mval = (mask > 0) & np.isfinite(dtb)
        dtb = dtb - (dtb[mval].mean() if np.any(mval) else 0.0)

        f_dtb = nmt.NmtField(mask, [dtb])                                    # beam=None (no pixwin)
        cl_coup_dtb = nmt.compute_coupled_cell(f_dtb, f_dtb)
        cl_dtb = w.decouple_cell(cl_coup_dtb)[0]

   
        tb = TB_mK.copy()
        val = (tb != hp.UNSEEN) & np.isfinite(tb) & (mask > 0)
        tb = np.where(val, tb - tb[val].mean(), 0.0)

        f_tb = nmt.NmtField(mask, [tb])                                     # beam=None (no pixwin)
        cl_coup_tb = nmt.compute_coupled_cell(f_tb, f_tb)
        cl_tb = w.decouple_cell(cl_coup_tb)[0]

        fsky = fsky_eff(mask)
        sigC_dtb = knox_sigma(ell_b, cl_dtb, BIN_WIDTH, fsky)
        sigC_tb  = knox_sigma(ell_b, cl_tb,  BIN_WIDTH, fsky)
        
        Dl_dtb = ell_b*(ell_b+1)*cl_dtb/(2*np.pi)
        Dl_tb  = ell_b*(ell_b+1)*cl_tb /(2*np.pi)
        Dlerr_dtb = ell_b*(ell_b+1)*sigC_dtb/(2*np.pi)
        Dlerr_tb  = ell_b*(ell_b+1)*sigC_tb /(2*np.pi)




        np.savez_compressed(
            os.path.join(out_dir, f"{shell_name}_dTB_cl.npz"),
            ell=ell_b, cl=cl_dtb, Dl=Dl_dtb,
            field="DTB", nside=NSIDE, lmax=LMAX,
            bin_width=BIN_WIDTH, shell_in=int(shell_in), shell_out=int(shell_out)
        )
        np.savez_compressed(
            os.path.join(out_dir, f"{shell_name}_TB_cl.npz"),
            ell=ell_b, cl=cl_tb, Dl=Dl_tb,
            field="TB_mK_minus_mean", nside=NSIDE, lmax=LMAX,
            bin_width=BIN_WIDTH, shell_in=int(shell_in), shell_out=int(shell_out)
        )


        plt.figure(figsize=(5.8, 4.2), dpi=200)
        m = np.isfinite(Dl_tb) & np.isfinite(Dlerr_tb) & (Dl_tb > 0) & (Dlerr_tb > 0)
        yerr_low  = np.minimum(Dlerr_tb[m], 0.99 * Dl_tb[m])   
        yerr_high = Dlerr_tb[m]

        plt.errorbar(ell_b[m], Dl_tb[m],
             yerr=np.vstack([yerr_low, yerr_high]),
             fmt='o', ms=3, lw=1, capsize=2, 
             label=r'$Gaussian errors')


        plt.xscale("log"); plt.yscale("log")
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$D_\ell $  [mK$^2$]")
        plt.grid(True, which='major', alpha=0.25)
       
        plt.title(f'{shell_name}  Δℓ={BIN_WIDTH} z= 1')
        plt.legend(frameon=False, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{shell_name}_Cl_binned.png"))
        plt.close()

