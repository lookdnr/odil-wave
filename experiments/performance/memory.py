import resource


def peak_rss(n_workers, baseline=0):
    KB = 1024
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * KB
    child_rss = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * KB
    )  # max single child
    own = max(child_rss - baseline, 0)  # child's own allocations excluding parents RSS
    return self_rss + n_workers * own, self_rss, child_rss, baseline
