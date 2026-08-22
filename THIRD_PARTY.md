# Third-party provenance

## thirdorder

MLFCS references and draws on the algorithms and workflow of the
[thirdorder](https://gitlab.com/sousaw/thirdorder) project, including its symmetry reduction of
finite-displacement third-order force constants, periodic-image geometry conventions, and the
ordered `sow`/`reap` workflow used to produce ShengBTE-compatible data.

MLFCS develops this lineage into a different, order-parameterized ASE/JAX architecture. New work
in MLFCS includes a common path from second to arbitrary order, recursive central-difference
stencils, sparse cluster storage, constrained acoustic-sum-rule solvers, CPU/GPU tensor
operations, calculator-independent APIs, and multiple export formats. These additions do not
erase or replace the attribution owed to thirdorder.

The thirdorder source consulted for development was revision
`7cb4ef0d2e036941165b016ba1b4f23bdd0e81c7`. Its source notices identify the following copyright
holders:

- Copyright (C) 2012–2018 Wu Li
- Copyright (C) 2012–2018 Jesús Carrete Montaña
- Copyright (C) 2012–2018 Natalio Mingo Bisquert
- Copyright (C) 2014–2018 Antti J. Karttunen
- Copyright (C) 2016–2018 Genadi Naydenov

thirdorder is licensed under the GNU General Public License v3.0 or later. MLFCS uses the same
license family; see [LICENSE](LICENSE). MLFCS does not claim authorship of thirdorder. Conversely,
the thirdorder authors are not represented as authors of the new MLFCS-specific contributions.

## Scientific reference data

Redistributed third-party test inputs and reference data retain their upstream notices and
licenses beside the files. Generated fixtures record upstream revisions, hashes, and regeneration
procedures in their local README files. A scientific citation does not replace compliance with
the corresponding software or data license.
