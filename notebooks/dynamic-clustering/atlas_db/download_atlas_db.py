import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool


ATLAS_DIR = "../data_sources/atlas/"
os.makedirs(ATLAS_DIR, exist_ok=True)
num_workers = 20


def get_request_url(pdb_id_name):
   return f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/{pdb_id_name}"

def get_request_command(pdb_id_name):
    request_url = get_request_url(pdb_id_name)
    command = "curl --silent -X GET " + request_url + f" -H accept: */* --output-dir {ATLAS_DIR}/{pdb_id_name} >> {ATLAS_DIR}/{pdb_id_name}/{pdb_id_name}.zip"
    return command

def download_atlas_traj(pdb_id_name):
    os.makedirs(f"{ATLAS_DIR}/{pdb_id_name}", exist_ok=True)
    os.system(get_request_command(pdb_id_name))


def run_jobs(job_list):
    if num_workers > 1:
        p = Pool(num_workers)
        p.__enter__()
        __map__ = p.imap_unordered
    else:
        __map__ = map

    for _ in tqdm(__map__(download_atlas_traj, job_list), total=len(job_list)):
        pass


if __name__ == "__main__":
    # Load the list of pdb ids
    atlas_file_list = pd.read_csv("./data/ATLAS.csv")
    jobs = []
    for pdb_id in atlas_file_list["PDB"].tolist():
        pdb_id_name = pdb_id.lower()
        pdb_id_name = pdb_id_name[:-1] + "_" + pdb_id_name[-1].upper()
        jobs.append(pdb_id_name)
    run_jobs(jobs)
