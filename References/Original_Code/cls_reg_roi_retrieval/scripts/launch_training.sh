#!/usr/bin/env bash
set -eu
make install
python -m cls_reg_roi_retrieval.cli.train_cli --epochs 20 "$@"
