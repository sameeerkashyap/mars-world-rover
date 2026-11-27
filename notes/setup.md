# Runtime setup

Keep simulator and reference JEPA dependencies in separate environments.
The simulator and rover training setup use Python 3.12. The legacy JEPA
reference setup uses Python 3.10. The root project metadata is not an
installation lock for these environments.

Source revisions are pinned in [upstream.json](upstream.json). The simulator
installer is `deploy/nautilus/runtime-setup.sh`; the rover model installer is
`deploy/nautilus-jepa-rover/setup.sh`. Use the matching deployment guide before
installing packages or launching a workload.

Save hardware details, package exports, launch commands, source hashes, logs,
and per-run test reports in the run artifact directory outside Git. On
Nautilus, setup output belongs under `/data/runs/setup/<run>/`. Keep local
copies in ignored `runs/` or `.cache/` storage.

See [getting started](../docs/getting-started.md) for commands and
[consolidated results](../docs/results.md) for completed findings. Historical
setup attempts and diagnostic reports remain in external artifact storage.
