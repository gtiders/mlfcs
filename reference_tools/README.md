# Scientific reference maintenance tools

English | [中文](README_ZH.md)

These maintainer scripts can regenerate scientific-reference fixtures and external sow plans
under `tests/reference/` when the required local inputs are available. The generated reference
data are not included in this repository. These scripts are not public MLFCS APIs or ordinary
user tools.

Each script documents its required local inputs. Run it only when deliberately updating a
reference dataset, then review provenance, checksums, atom ordering, numerical tolerances, and
redistribution terms before handling the result. Normal users and ordinary CI do not need to
run these scripts.
