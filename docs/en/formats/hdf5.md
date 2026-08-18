# Native HDF5 v2

Native HDF5 v2 is the only MLFCS interchange schema. It preserves sparse support explicitly and
does not infer support from zero-valued tensors. Dense materialization is only performed by a
target writer that requires it.

Files from the pre-v2 schema are rejected with an unsupported-schema error. There is no migration
reader that guesses old atom-order semantics.
