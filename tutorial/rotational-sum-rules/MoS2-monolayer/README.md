# MoS2 monolayer

The two independent FC2 fits use the same $8\times8\times1$ supercell, 8 Å cutoff, and
two-body cluster space. The training trajectory already contains displaced
ASE structures and their stored forces. `asr/fit.py` applies ASR only;
`born-huang-huang/fit.py` applies ASR, Born–Huang, and Huang constraints.

The band plot compares the resulting Phonopy FC2 text files. Run both fit
scripts before `plot.py`.

For the supplied trajectory, the rotational projection keeps the training
relative force error near 10.05%. It retains the two rotational directions
resolvable above the measured geometry error while preserving the separate
ASR result. The reported Born–Huang and Huang residuals are small but not
exactly zero; this is not a claim of strict rotational invariance.
