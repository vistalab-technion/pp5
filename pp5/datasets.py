import logging
import multiprocessing as mp
import multiprocessing.pool
import os
import shutil
import tempfile
import time
from pathlib import Path

import pp5
from pp5.external_dbs import pdb
from pp5.external_dbs.pdb import PDBQuery
from pp5.protein import ProteinRecord, ProteinInitError
from pp5.align import structural_align

import pandas as pd

LOGGER = logging.getLogger(__name__)


def worker_process_init(base_workers_dl_dir, *args):
    # Provide each process with a unique download folder.
    pid = os.getpid()
    worker_dl_dir = base_workers_dl_dir.joinpath(f'{pid}')
    os.makedirs(worker_dl_dir, exist_ok=True)
    LOGGER.info(f'Worker process {pid} using dir {worker_dl_dir}...')
    # Override the default with the process-specific dir to have a
    # different download directory for each process.
    pp5.BASE_DOWNLOAD_DIR = worker_dl_dir


def clean_worker_downloads(base_workers_dl_dir):
    n_copied = 0
    downloaded_files = base_workers_dl_dir.glob('**/*/*.*')
    for downloaded_file in downloaded_files:
        rel_path = downloaded_file.relative_to(base_workers_dl_dir)
        # Remove pid dir and append the relative path to the main data dir
        data_path = pp5.BASE_DATA_DIR.joinpath(*rel_path.parts[1:])
        if data_path.is_file():
            continue
        # The downloaded file is not in the data dir, so copy it.
        shutil.copy2(str(downloaded_file), str(data_path))
        n_copied += 1
    LOGGER.info(f'Copied {n_copied} downloaded files from '
                f'{base_workers_dl_dir} into {pp5.BASE_DATA_DIR}')
    shutil.rmtree(base_workers_dl_dir)
    LOGGER.info(f'Deleted temp folder {base_workers_dl_dir}')


def collect_precs(query: PDBQuery):
    pdb_ids = query.execute()
    LOGGER.info(f"Got {len(pdb_ids)} structures from PDB")

    base_workers_dl_dir = Path(tempfile.gettempdir()).joinpath('pp5_data')
    os.makedirs(str(base_workers_dl_dir), exist_ok=True)

    async_results = []
    with mp.pool.Pool(processes=8,
                      initializer=worker_process_init,
                      initargs=(base_workers_dl_dir,)) as pool:
        for i, pdb_id in enumerate(pdb_ids):
            r = pool.apply_async(ProteinRecord.from_pdb, args=(pdb_id,),
                                 kwds=dict(dihedral_est_name='erp'))
            async_results.append(r)

        start_time, counter = time.time(), 0
        for async_result in async_results:
            try:
                prec = async_result.get(30)
                counter += 1
                pps = counter / (time.time() - start_time)
            except TimeoutError as e:
                LOGGER.error("Timeout getting async result, skipping")
            except ProteinInitError as e:
                LOGGER.error(f"Failed to create protein: {e}")
            except Exception as e:
                LOGGER.error("Unexpected error", exc_info=e.__cause__)

            # Write to file
            prec.to_csv(pp5.data_subdir('proteins'))
            if counter % 10 == 0:
                LOGGER.info(f'Total collected: {counter} ({pps:.1f} p/sec)')

        clean_worker_downloads(base_workers_dl_dir)
        LOGGER.info(f"Done: {counter} proteins collected ({pps:.1f} p/sec).")


def collect_protein_groups(ref_pdb_id: str):
    """

    :param ref_pdb_id: Reference PDB ID with chain.
    :return:
    """
    ref_prec = ProteinRecord.from_pdb(ref_pdb_id)

    blast_query = pdb.PDBCompositeQuery(
        pdb.PDBExpressionSystemQuery('Escherichia Coli'),
        pdb.PDBResolutionQuery(max_res=1.8),
        pdb.PDBSequenceQuery(pdb_id=ref_pdb_id, e_cutoff=1.)
    )

    # BLAST queries return entity ids
    pdb_entities = blast_query.execute()
    LOGGER.info(f'Got {len(pdb_entities)} entities from PDB')

    def add_prec(d: dict, prec: ProteinRecord, struct_rmse: float,
                 ref_group: bool):
        d.setdefault('unp_id', []).append(prec.unp_id)
        d.setdefault('pdb_id', []).append(prec.pdb_id)
        d.setdefault('struct_rmse', []).append(struct_rmse)
        d.setdefault('description', []).append(prec.pdb_meta.description)
        d.setdefault('src_org', []).append(prec.pdb_meta.src_org)
        d.setdefault('src_org_id', []).append(prec.pdb_meta.src_org_id)
        d.setdefault('host_org', []).append(prec.pdb_meta.host_org)
        d.setdefault('host_org_id', []).append(prec.pdb_meta.host_org_id)
        d.setdefault('ref_group', []).append(ref_group)

    groups_data = {}
    add_prec(groups_data, ref_prec, 0.0, True)

    for pdb_entity in pdb_entities:
        pdb_id, entity_id = pdb_entity.split(':')
        entity_id = int(entity_id)
        try:
            prec = ProteinRecord.from_pdb_entity(pdb_id, entity_id)
        except ProteinInitError as e:
            LOGGER.error(f"Failed to create protein: {e}")

        # Run structural alignment between the ref and current structure
        rmse, _ = structural_align(ref_pdb_id, prec.pdb_id,
                                   rmse_cutoff=2., min_stars=50)
        if not rmse or rmse > 2:
            LOGGER.info(f'Rejecting {pdb_id} due to insufficient structural '
                        f'similarity, RMSE={rmse}')
            continue

        add_prec(groups_data, prec, rmse, prec.unp_id == ref_prec.unp_id)

    df = pd.DataFrame(groups_data)
    df.sort_values(by=['ref_group', 'unp_id'], ascending=False, inplace=True)
    out_file = pp5.out_subdir('protein_groups')\
        .joinpath(f'{ref_pdb_id.replace(":", "_")}.csv')
    df.to_csv(out_file)
    return df


if __name__ == '__main__':
    # Query PDB for structures
    # query = pdb.PDBCompositeQuery(
    #     pdb.PDBExpressionSystemQuery('Escherichia Coli'),
    #     pdb.PDBResolutionQuery(max_res=1.0)
    # )
    #
    # collect_precs(query)

    df = collect_protein_groups('1MWC:A')
    print(df)
