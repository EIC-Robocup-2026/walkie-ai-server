import os, sys, glob
import numpy as np
sys.path.insert(0, os.getcwd())
from services.grasp.providers.graspnet_provider import GraspNetProvider

cfg = {
    "graspnet_root": "~/graspnet-baseline",
    "checkpoint_path": "~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar",
    "device": "",
    "num_point": 10000, "voxel_size": 0.005,
    "snapshot_inputs": False,
    "pitch_offset_deg": 0, "roll_offset_deg": 0, "yaw_offset_deg": 0,  # see RAW approach=col0
}
prov = GraspNetProvider(cfg)
prov.load_model()

f = sorted(glob.glob("test_clouds/*_raw.npy"))[0]
cloud = np.load(f).astype(np.float32)
up = np.array([0.0,-1.0,0.0])
print("\n===== FILE", os.path.basename(f), "N=", len(cloud), "=====")

def show(tag, grasps, n=8):
    print(f"\n--- {tag}: {len(grasps)} grasps ---")
    for g in grasps[:n]:
        R = np.array(g["rotation"]); appr = R[:,0]; clos = R[:,1]
        # angle of approach from horizontal: approach . up
        au = float(appr@up); cu = float(clos@up)
        print(f"  score={g['score']:.3f} anti={g['antipodal_score']} w={g['width']*100:5.1f}cm "
              f"appr={np.round(appr,2)} (appr.up={au:+.2f}) clos.up={cu:+.2f}")

# A) raw graspnet, no preference, no antipodal, no filters
gA = prov.infer(cloud, score_threshold=0.0, max_grasps=10, antipodal=False,
                approach_preference="none", outlier_removal=True)
show("A raw graspnet (no pref, no antipodal)", gA)

# B) just side preference soft, no hard drops (max_approach_up=1, max_closing_up=1)
gB = prov.infer(cloud, score_threshold=0.0, max_grasps=10, antipodal=False,
                approach_preference="side", up=up.tolist(),
                max_approach_up=1.0, max_closing_up=1.0, center_weight=0, closing_weight=0)
show("B side soft only (no hard drops)", gB)

# C) the user's actual config: side + antipodal + hard drops
gC = prov.infer(cloud, score_threshold=0.0, max_grasps=10, antipodal=True,
                voxel_size=0.003, approach_preference="side", up=up.tolist(),
                max_approach_up=0.2, max_closing_up=0.4, center_weight=0.8, closing_weight=1.0)
show("C user config (side+antipodal+hard drops)", gC)
