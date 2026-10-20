# Readiness-To-Train on VSC

Shared infrastructure reference: [VSC Library](../vsc-library/README.md). The parent Git commit pins its revision; inspect it with `git submodule status -- vsc-library`.
Its [official repository snapshot](<../vsc-library/VSC Repository.txt>),
[guides](../vsc-library/guides/) and [Slurm templates](<../vsc-library/slurm templates/README.md>)
are read-only. Keep this project's actual paths, environment, resource choices,
submission/resume steps and run evidence in this file or the project's own run logs.
Check [upstream review flags](../vsc-library/UPSTREAM_CHANGES.md) and the
[live Leuven status](https://status.vscentrum.be/tier2_leuven.html) before applying a changed policy.

This checkout had no Slurm job scripts in the inspected source. Adding this
reference does not establish a tested HPC deployment or require moving the
existing workflow to VSC. Before a first cluster run, record the approved data
location, environment, preprocessing/compute command, account, resource needs,
output path and resume/transfer procedure here. Use a bounded pilot first.
