from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import islice
import os


def worker_count(requested=None):
    available = os.cpu_count() or 1
    return max(1, min(8, available, int(requested) if requested is not None else available))


@contextmanager
def tile_results(tiles, process, check, workers):
    """Process a bounded number of tiles in parallel and yield them in write order."""
    if workers == 1 or len(tiles) == 1:
        yield ((tile, process(tile)) for tile in tiles)
        return

    pool = ThreadPoolExecutor(max_workers=min(workers, len(tiles)))

    def results():
        remaining = iter(tiles)
        pending = deque((tile, pool.submit(process, tile)) for tile in islice(remaining, 2 * workers))
        while pending:
            check()
            tile, future = pending.popleft()
            yield tile, future.result()
            next_tile = next(remaining, None)
            if next_tile is not None:
                pending.append((next_tile, pool.submit(process, next_tile)))

    try:
        yield results()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
