"""
MEASURED launch configurations -- GENERATED FILE, do not hand-edit.

Produced by `experiments/gpu_m22_autotune.py`, which prints the whole search it ran. Regenerate it
on a different GPU rather than editing a number: the point of the table is that every entry was
timed on the machine that uses it.

    NVIDIA GeForce RTX 5090 D v2, torch 2.11.0+cu128, triton 3.6.0

ONLY DEVIATIONS FROM THE DEFAULT ARE LISTED. Every other shape uses `autotune.DEFAULT`, because the
measurement did not justify anything else: candidates for a short row differ by a few percent run to
run, and a candidate has to beat the default by more than `autotune.NOISE` (5%) to be
recorded here. Where the choice does matter it is not marginal -- one of the rows below is a single
row of 262144 columns, where the default is several times slower.

Keys are `(rows class, ncols bucket)`; see `neptune_codegen/autotune.py` for what a bucket is, why
the class exists, and why a choice has to clear the noise margin.
"""
TABLE = {
    ('few', 4096): (512, 2),
    ('many', 1024): (512, 2),
    ('many', 262144): (8192, 8),
    ('single', 262144): (8192, 8),
}
