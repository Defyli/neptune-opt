"""
MEASURED launch configurations -- GENERATED FILE, do not hand-edit a number.

Add or refresh ONE machine with:

    python experiments/measure_launch_configs.py

which prints every candidate timing it took and merges the result in, leaving the other machines
alone. Regenerate on the machine that will use the table: the table does NOT transfer between GPUs
(measured: four deviations from the default on an RTX 5090 D v2, one on an RTX 4090 D, agreeing only
on the single-long-row case). A machine with no entry here uses `autotune.DEFAULT`, which is what
every measurement before I6 used.

ONLY DEVIATIONS FROM THE DEFAULT ARE LISTED. Every other shape uses `autotune.DEFAULT`, because the
measurement did not justify anything else: candidates for a short row differ by a few percent run to
run, and a candidate has to beat the default by more than `autotune.NOISE` (5%) to be
recorded. Where the choice does matter it is not marginal -- a single row of 262144 columns is 3x
faster with the configuration recorded for it.

Keys inside `table` are `(rows class, ncols bucket)`; see `neptune_codegen/autotune.py` for what a
bucket is, why the class exists, and why a choice has to clear the noise margin.
"""
TABLES = {
    'rtx-4090-d': {
        'measured_on': 'NVIDIA GeForce RTX 4090 D, torch 2.11.0+cu128, triton 3.6.0',
        'noise': 0.05,
        'table': {
            ('single', 262144): (8192, 8),
        },
    },
    'rtx-5090-d-v2': {
        'measured_on': 'NVIDIA GeForce RTX 5090 D v2, torch 2.11.0+cu128, triton 3.6.0',
        'noise': 0.05,
        'table': {
            ('few', 4096): (512, 2),
            ('many', 1024): (512, 2),
            ('many', 262144): (8192, 8),
            ('single', 262144): (8192, 8),
        },
    },
    'v100-pcie-32gb': {
        'measured_on': 'Tesla V100-PCIE-32GB, torch 2.0.1+cu117, triton 2.0.0',
        'noise': 0.05,
        'table': {
            ('many', 262144): (4096, 4),
            ('single', 262144): (8192, 8),
        },
    },
}
