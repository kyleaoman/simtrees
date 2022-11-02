from simtrees._setup_cfg import paths
from simfiles.configs.APOSTLE import (
    res_str,
    vol_str,
    phys_str,
    snap_id,
    machine,
)
from itertools import product

path_bases = {
    "cosma": "/cosma6/data/dp004/lg/snapshots_all/Merger_Trees",
}

snap = 127
for res, vol, phys in product(range(1, 4), range(1, 13), ["hydro", "DMO"]):
    if (res == 1) and (
        ((phys == "hydro") and (vol not in [1, 4, 6, 10, 11]))
        or ((phys == "DMO") and (vol not in [1, 4, 11]))
    ):
        continue

    fpath = "{:s}/{:s}_{:s}_{:s}/snapshots/treedir_{:03d}/".format(
        path_bases[machine], vol_str[vol], res_str[res], phys_str[phys], snap
    )
    fbase = "tree_{:03d}".format(snap)
    sfbase = "subfind_{:03d}".format(snap)
    paths[snap_id(res=res, phys=phys, vol=vol, snap=snap)] = (
        fpath,
        fbase,
        sfbase,
    )
