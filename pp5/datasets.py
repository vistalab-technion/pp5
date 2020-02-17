import logging
import multiprocessing as mp
import time

from pp5.external_dbs import pdb
from pp5.protein import ProteinRecord, ProteinInitError

LOGGER = logging.getLogger(__name__)


def collect_data():
    # Query PDB for structures
    query = pdb.PDBCompositeQuery(
        pdb.PDBExpressionSystemQuery('Escherichia Coli'),
        pdb.PDBResolutionQuery(max_res=1.0)
    )

    pdb_ids = query.execute()
    LOGGER.info(f"Got {len(pdb_ids)} structures from PDB")

    async_results = []
    with mp.pool.Pool(processes=8) as pool:
        for i, pdb_id in enumerate(pdb_ids):
            async_results.append(
                pool.apply_async(ProteinRecord.from_pdb, (pdb_id,))
            )

        start_time, counter = time.time(), 0
        for async_result in async_results:
            try:
                protein_recs = async_result.get(30)
            except TimeoutError as e:
                LOGGER.error("Timeout getting async result, skipping")
            except ProteinInitError as e:
                LOGGER.error(f"Failed to create protein: {e}")
            except Exception as e:
                LOGGER.error("Unexpected error", exc_info=e.__cause__)

            counter += len(protein_recs)
            pps = counter / (time.time() - start_time)
            LOGGER.info(f'Collected {protein_recs} ({pps:.1f} proteins/sec)')

            # TODO: Write to file

        LOGGER.info(f"Done: {counter} proteins collected.")


if __name__ == '__main__':
    collect_data()
