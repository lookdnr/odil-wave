import resource


def peak_rss(n_workers):
    KB = 1024
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * KB
    child_rss = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * KB
    )  # max single child
    return self_rss + n_workers * child_rss, self_rss, child_rss
