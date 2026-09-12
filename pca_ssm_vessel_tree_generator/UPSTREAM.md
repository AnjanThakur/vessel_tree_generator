# Upstream provenance

- Repository: https://github.com/Gauravch-dev/pca-ssm_vessel_tree_generator.git
- Imported commit: `acc98bc`
- Imported on: 2026-08-04

The upstream `.git` directory was removed so this folder is tracked as part of the parent repository rather than as an accidental nested repository or submodule.

Integration cleanup:

- script paths are resolved relative to each script instead of the caller's working directory;
- obsolete exploratory `test.py` and `test.ipynb` files with a developer-specific absolute path were excluded;
- a README and dependency list were added.
