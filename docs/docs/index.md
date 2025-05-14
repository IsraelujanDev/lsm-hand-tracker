# lsm_hand_tracker documentation!

## Description

An exploratory project to generate a Database of pictures about the LSM

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://Not defined yet/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://Not defined yet/data/` to `data/`.


