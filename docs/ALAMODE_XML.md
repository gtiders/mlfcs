# ALAMODE XML output

[中文](ALAMODE_XML_ZH.md)

MLFCS writes second- through fourth-order force constants into one ALAMODE FCSXML document:

```python
force_constants.write("force_constants.xml", format="alamode")
force_constants.write("fc3.xml", format="alamode", order=3)
```

Omitting `order` writes every available order among FC2, FC3, and FC4. Higher orders remain
available through MLFCS HDF5 because the ALAMODE FCSXML schema implemented here defines only
`HARMONIC`, `ANHARM3`, and `ANHARM4` sections.

## Atom order and primitive mapping

The exported `Structure/Position` sequence is exactly `force_constants.supercell`; it is not
sorted by element, primitive atom, or translation. MLFCS then writes the mapping explicitly:

- `primitive_index[i]` selects the primitive atom represented by supercell atom `i`;
- `cell_translation[i]` identifies its integer supercell translation;
- `Symmetry/Translations/map` records the resulting primitive/translation-to-supercell lookup.

Consequently, users can control the atom order by constructing the MLFCS supercell in the desired
order while keeping those two metadata arrays consistent. The exporter validates that the mapping
is complete and one-to-one. It never reruns spglib and never silently changes the mapping.

FCSXML force-constant pairs use the ALAMODE convention: the first atom is a primitive-cell atom,
the remaining atoms are exact supercell indices, and their final integer is a periodic mirror-cell
identifier. This is a format representation, not an additional atom permutation.

## Periodic images and units

ALAMODE numbers a fixed list of 27 mirror cells: the central cell followed by all translations in
`{-1, 0, 1}^3`. Degenerate closest images are emitted separately and the force constant is divided
by their multiplicity, matching the official Python ALM writer. Repeated occurrences of the same
atom share one mirror choice. MLFCS additionally verifies the result against ASE's general minimum
image; if the fixed 27-image representation is insufficient, export first tries an integral
unimodular Minkowski rebase of the target supercell. This preserves the physical supercell
lattice, atoms, and IFCs while often making a non-reduced representation encodable by the
27-image format. Only a geometry that remains unrepresentable after that rebase is rejected.

MLFCS stores order-`n` force constants in eV/angstrom^`n`. FCSXML values are written in
Ry/bohr^`n` using `value * bohr^n / Ry`. Lattice vectors are written in bohr and fractional
positions are unchanged.

## Provenance and compatibility note

The layout and mirror expansion are adapted from the MIT-licensed `alm.fcsxml.Fcsxml` writer in
`ttadano/ALM` revision `f1d668f210d3e95355643132144f3fd1ec10d4d7`; its complete attribution and
license are embedded at the top of `src/mlfcs/io/alamode.py`. Semantic tests cover multicomponent ordering,
translation maps, mirror degeneracy, repeated atoms, unit conversion, and FC2--FC4 sections.

ALAMODE's separate Python XML reader may choose an origin-shifted primitive cell when it invokes
spglib and can reject some otherwise valid XML files—including files produced by ALM's own Python
writer. MLFCS deliberately does not imitate that origin-sensitive rediscovery. The explicit map in
the XML remains the authority for downstream ALAMODE calculations.
