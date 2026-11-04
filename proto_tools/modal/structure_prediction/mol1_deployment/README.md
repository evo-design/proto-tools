# Mol-1 deployment

Serves [`mol1-prediction`](../../../tools/structure_prediction/mol1/README.md) with a
persistent worker and a shared model cache. The image warmup installs the isolated
environment, downloads the bundled checkpoint and reference data, and runs a short
prediction without MSA search.

Deploy with `proto-tools deploy --apps mol1 --env proto-env`, then select
`Mol1Config(device="modal")` to run on your Modal account.

GPU deployment and benchmarks have not yet been run. No runtime measurements are
claimed here; regenerate this page with `pytest --benchmark-toolkit=mol1 --use-modal
--benchmark-report=work/mol1-bench` after validating the deployment.
