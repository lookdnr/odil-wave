import psutil
import time


def peak_tree_rss(proc, poll=0.03):
    """Peak summed Resident Set Size (RAM in use) of proc + all descendants"""
    p = psutil.Process(proc.pid)  # identify proc
    peak = 0

    while proc.poll() is None:
        rss = 0
        for q in [p, *p.children(recursive=True)]:
            try:
                rss += q.memory_info().rss  # sum RSS
            except psutil.NoSuchProcess:
                pass
        peak = max(peak, rss)  # record largest
        time.sleep(poll)
    return peak
