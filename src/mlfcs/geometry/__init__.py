"""Lattice geometry and space-group actions.

This package is deliberately free of compiled dependencies so that importing it
stays cheap: the prepared lattice, lattice-site addresses and the site matcher
live in :mod:`mlfcs.geometry.lattice`, space-group actions in
:mod:`mlfcs.geometry.symmetry`, and the validated structure records with their
compiled quotient index in :mod:`mlfcs.geometry.structure`. Import the module
that owns a name instead of relying on package attributes.
"""
