# ruff: noqa
# Vendored from https://github.com/autosome-ru/parade @ f8e3e02 (predictor/model/legnet_classifier.py).
# Upstream this file is only an alias (`from legnet import *`), but PARADE's checkpoints pickle the
# model class as `legnet_classifier.LegNetClassifier` -- so this module must stay importable under
# that name for the checkpoint to unpickle. Kept byte-for-byte (this header is the only addition).
# Alias to 'legnet.py'

from legnet import *
