# Third-party provenance

## ALM FCSXML writer

The ALAMODE XML adapter is adapted from the pure-Python `alm.fcsxml.Fcsxml` writer in the
[ttadano/ALM](https://github.com/ttadano/ALM) repository at revision
`f1d668f210d3e95355643132144f3fd1ec10d4d7`. That implementation is copyright Terumasa Tadano
and distributed under the MIT License. MLFCS retains its XML layout, unit conversion, and
27-image closest-mirror convention while replacing primitive discovery with MLFCS-controlled
atom and translation mappings. The adapted module carries a source-level attribution notice.
The complete upstream license is retained in [third_party_licenses/ALM-MIT.txt](third_party_licenses/ALM-MIT.txt).

## Scientific reference data

Redistributed third-party test inputs and reference data retain their upstream notices and
licenses beside the files. Generated fixtures record upstream revisions, hashes, and regeneration
procedures in their local README files. A scientific citation does not replace compliance with
the corresponding software or data license.
