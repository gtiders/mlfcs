# Graphene

The two independent FC2 fits use the same $7\times7\times1$ supercell, 8 Å cutoff, and
two-body cluster space. The stored training file contains a displacement array
and forces; each fit script reconstructs an ASE `Atoms` snapshot and attaches
those stored forces through ASE's `SinglePointCalculator`. `asr/fit.py` applies
ASR only; `born-huang-huang/fit.py` applies ASR, Born–Huang, and Huang
constraints.

The band plot compares the resulting Phonopy FC2 text files. Run both fit
scripts before `plot.py`.
