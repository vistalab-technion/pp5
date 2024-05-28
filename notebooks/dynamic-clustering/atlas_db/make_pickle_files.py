import os
import mdtraj
import pickle

from tqdm import tqdm
from pathlib import Path
from protein import get_pdb_angles, flatten_md_data
from multiprocessing import Pool


MD_DATA = Path("/mnt/walkure_public/users/sanketh/atlas_db/")
out_folder = Path("./data_atlas/")
os.makedirs(out_folder, exist_ok=True)
num_workers = 64

def main():
    jobs = []
    for name in os.listdir(MD_DATA):
        #if os.path.exists(f'{args.outdir}/{name}.npz'): continue
        jobs.append(name)

    if num_workers > 1:
        p = Pool(num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map

    for _ in tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass

    if num_workers > 1:
        p.__exit__(None, None, None)

def do_job(sim_name: str):
    # PDB file
    pdb_file = MD_DATA / sim_name / f"{sim_name}.pdb"
    # Trajectory file
    traj_file = MD_DATA / sim_name / f"{sim_name}_prod_R1_fit.xtc"

    # Load the MD file
    print(f"Loading {sim_name}...")
    traj = mdtraj.load(str(traj_file), top=str(pdb_file))
    static_data, md_angles = flatten_md_data(traj)
    coordinates = traj.xyz[:, traj.topology.select("backbone"), :].reshape(
        traj.xyz.shape[0], traj.topology.select("backbone").shape[0]//4, 4, 3
    )
    assert traj.topology.select("backbone").shape[0]//4 == static_data.shape[0] + 2
    # with open(MD_DATA / sim_name / f"{sim_name}_processed_R1.pkl", "wb") as f:
    with open(out_folder / f"{sim_name}_processed_R1.pkl", "wb") as f:
        pickle.dump((static_data, coordinates, md_angles), f)
        f.close()
    print(f"Saved {sim_name}...")

main()
