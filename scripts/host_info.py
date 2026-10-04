"""Host facts recorded with fits and scores (code check CR-02, CR-05). Standard library only."""


def cpu_info(path="/proc/cpuinfo"):
    """(model name, highest of avx512f / avx2 / sse4_2 in the CPU flags, or 'none') of the first CPU listed in
    /proc/cpuinfo; raises if the file has no 'model name' or 'flags' line (no silent blank)."""
    model = flags = None
    with open(path) as f:
        for line in f:
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "model name" and model is None:
                model = value.strip()
            elif key == "flags" and flags is None:
                flags = set(value.split())
            if model is not None and flags is not None:
                break
    if model is None or flags is None:
        raise RuntimeError(f"{path} has no 'model name' or 'flags' line")
    return model, next((x for x in ("avx512f", "avx2", "sse4_2") if x in flags), "none")
