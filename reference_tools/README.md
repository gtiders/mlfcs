# Scientific reference maintenance tools

These scripts regenerate frozen scientific-reference fixtures and external sow plans under
`tests/reference/`. They are maintainer utilities, not public MLFCS APIs or ordinary user tools.

Each script documents its required local inputs. Run it only when deliberately updating a
reference dataset, then review provenance, checksums, atom ordering, numerical tolerances, and
redistribution terms before committing the result. Normal users and ordinary CI do not need to
run these scripts.
