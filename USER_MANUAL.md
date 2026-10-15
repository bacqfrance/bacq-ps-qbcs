# BACQ-PS-QBCS — Error-Mitigated BCS Quench Benchmark

User manual for running the benchmark on a **local Qiskit/Aer simulator** or on a
**real QPU**: IBM (via [IBM Quantum Platform](https://quantum.cloud.ibm.com/)), IQM (via
[IQM Resonance](https://resonance.iqm.tech/)), IQM [Euro-Q-Exa](https://www.eurohpc-ju.europa.eu/eurohpc-quantum-computers/our-quantum-computers_en#euro-q-exa) (EQE1) at
[LRZ](https://www.lrz.de/en/), IQM [VLQ](https://www.eurohpc-ju.europa.eu/eurohpc-quantum-computers/our-quantum-computers_en#vlq) at [IT4I](https://www.it4i.cz/en) and AQT
[PIAST-Q](https://www.eurohpc-ju.europa.eu/eurohpc-quantum-computers/our-quantum-computers_en#piast-q) at [PCSS](https://www.psnc.pl/).

Based on:

> *Mitigating crosstalk errors by randomized compiling: Simulation of the BCS model
> on a superconducting quantum computer*
> H. Perrin, T. Scoquart, A. Shnirman, J. Schmalian, K. Snizhko, [Phys. Rev. Research **6**, 013142 (2024)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.013142), [arXiv:2305.02345](https://arxiv.org/abs/2305.02345).

---

## Contents

- [BACQ-PS-QBCS — Error-Mitigated BCS Quench Benchmark](#bacq-ps-qbcs--error-mitigated-bcs-quench-benchmark)
  - [Contents](#contents)
  - [1. Introduction](#1-introduction)
  - [2. Layout](#2-layout)
  - [3. Setup](#3-setup)
    - [3.1 Software modules](#31-software-modules)
    - [3.2 Build the base virtual environment](#32-build-the-base-virtual-environment)
      - [Setup (Windows)](#setup-windows)
      - [Setup (Linux)](#setup-linux)
  - [4. Quickstart — run the benchmark on the local simulator](#4-quickstart--run-the-benchmark-on-the-local-simulator)
    - [4.1 Circuit caches and format](#41-circuit-caches-and-format)
    - [4.2 Local simulator with a real backend's noise model](#42-local-simulator-with-a-real-backends-noise-model)
    - [4.3 Minimal script sequence — build every circuit the QPU runs need](#43-minimal-script-sequence--build-every-circuit-the-qpu-runs-need)
  - [5. The pipeline — `bacq_ps_qbcs`](#5-the-pipeline--bacq_ps_qbcs)
    - [5.1 Constructor](#51-constructor)
    - [5.2 `build(circuit_transpilation_LNN=False)`](#52-buildcircuit_transpilation_lnnfalse)
    - [5.3 `submit()` signature](#53-submit-signature)
    - [5.4 Circuit cache, qubit layout and run names](#54-circuit-cache-qubit-layout-and-run-names)
    - [5.5 Idempotency — the pipeline is a chain of checkpoints](#55-idempotency--the-pipeline-is-a-chain-of-checkpoints)
      - [The checkpoints](#the-checkpoints)
      - [Shared vs per-run](#shared-vs-per-run)
      - [Forcing a stage to re-run](#forcing-a-stage-to-re-run)
  - [6. Configuration knobs that matter for a benchmark run](#6-configuration-knobs-that-matter-for-a-benchmark-run)
  - [7. Running on a real QPU](#7-running-on-a-real-qpu)
  - [8. Outputs](#8-outputs)
  - [9. Validation / tests](#9-validation--tests)
    - [9.1 Your circuit caches are safe during `pytest source/tests/`](#91-your-circuit-caches-are-safe-during-pytest-sourcetests)
  - [10. Post-processing — plots and the summary deliverable](#10-post-processing--plots-and-the-summary-deliverable)
    - [10.1 Run them in this order](#101-run-them-in-this-order)
    - [10.2 The scripts](#102-the-scripts)
    - [10.3 Things that will trip you up](#103-things-that-will-trip-you-up)

---

## 1. Introduction

The pipeline simulates the non-equilibrium dynamics of a 3-qubit **BCS
(Bardeen–Cooper–Schrieffer) quench** — single-particle energies `ε = [-1, 0, 1]`,
pairing strength `g = 0.5` — via Trotter decomposition at a sweep of depths `Nt`,
and benchmarks how well the following error-mitigation strategies recover the exact
observables on noisy hardware:

| Strategy | Roles produced | Description |
|---|---|---|
| `NEC`     | `raw`, `raw_nec` | Noise Estimation Circuits   |
| `RC+NEC`  | `rc`, `rc_nec`   | Randomized Compiling + NEC |
| `cRC+NEC` | `crc`, `crc_nec` | Crosstalk-aware RC + NEC (twirls neighbour-qubit noise) |
| `SPAM`    | —                | State-Preparation-And-Measurement error correction (confusion-matrix aka readout error matrix) |

Any subset can be requested; the pipeline adapts automatically.

The circuit build/compile stages run in **myQLM**; execution runs through **Qiskit**
(local `AerSimulator` or a hardware provider). Only the build needs myQLM: it saves
every circuit as Qiskit QPY (plus the exact reference values) in a shared cache,
and submission, collection and analysis read only those files — so the real-QPU
environments don't need myQLM at all (§3.1, §5.4).

---

## 2. Layout

```
Quantum_Benchmarking_project/
├── source/                              # all code
│   ├── core/                            # the pipeline library, imported as source.core[.<subfolder>].<module>
│   │   ├── qbcs.py                      #   bacq_ps_qbcs — the main class: build, submit, collect, analyze
│   │   ├── health_check.py              #   QPU health check: build, submit, collect, require before a run — see §7
│   │   ├── common.py                    #   myQLM-free helpers: run/cache folder paths (circuits_root(), get_circuit_folder(), …)
│   │   ├── circuits/                    #   circuit construction (only main, circuit_functions, rc_functions load myQLM on import)
│   │   │   ├── main.py                  #   BCS Hamiltonian, myQLM plugins
│   │   │   ├── circuit_functions.py     #   BCS Trotter circuit builder
│   │   │   ├── rc_functions.py          #   rcCNOT / crcCNOT gate definitions
│   │   │   ├── building.py              #   the steps build() runs: BCS circuits, plugin batches, exact-observable cache, noise angles
│   │   │   ├── SPAM_functions.py        #   SPAM calibration + confusion matrix
│   │   │   └── circuit_drawings.py      #   circuit diagrams of a chunk's roles
│   │   ├── records/                     #   what a run saves, and how it is named
│   │   │   ├── run_names.py             #   builds and parses every run, plot and summary name — see §5.4
│   │   │   ├── run_info.py              #   run_info.json: what a run was (shots, samples, depths, …) — see §10.2
│   │   │   ├── job_records.py           #   job record files (write, archive failed attempts)
│   │   │   ├── health_check_records.py  #   health check files / job grouping helpers
│   │   │   ├── job_timing.py            #   per-job timing blocks (transpile/submit/QPU/queue) — see §7, §10.2
│   │   │   ├── timestamps.py            #   timestamp CSV logs (local + UTC) and circuit build times
│   │   │   ├── chunk_scope.py           #   which chunks a run covers / still lacks
│   │   │   └── circuit_check.py         #   executed-vs-submitted circuit check — see §7
│   │   ├── qpu/                         #   provider access and QPU data
│   │   │   ├── providers.py             #   constants the QPU scripts share (IQM Resonance URL, health-check shots, standalone health check), `job_state_of_status()` (done / failed / pending from a job-status text) and `ibm_batch()`, the IBM Batch + sampler that is always closed
│   │   │   ├── tokens.py                #   read_token() / token_path(): the provider tokens and access files in tokens/
│   │   │   ├── iqm_vlq.py               #   IQM VLQ (LEXIS/HEAppE) login, upload and calibration
│   │   │   ├── qpu_snapshot.py          #   calibration snapshot → offline backend + Aer noise model — see §7
│   │   │   ├── calibration_input.py     #   calibration_file= (provider format or snapshot) → snapshot + offline backend — see §5.4
│   │   │   ├── layout_selection.py      #   picks a run's physical qubits (optimized / random / lowest / given) — see §5.4
│   │   │   ├── aqt.py                   #   AQT PIAST-Q: QPU time per circuit, resilient job with retries and shot split
│   │   │   ├── remote_jobs.py           #   live job state, submit-error kinds and retries (shared by qbcs and health_check)
│   │   │   └── job_status_check.py      #   live job status vs. the records on disk
│   │   ├── runner_support/              #   shared by the scripts users run
│   │   │   ├── benchmark_args.py        #   the shared command line (benchmark size, backend=, layout=, calibration_file=)
│   │   │   ├── run_simulation.py        #   the shared steps of the simulate scripts (simulate_run(), after the calibration snapshot)
│   │   │   ├── run_local_simulation.py  #   the shared flow of the local simulation scripts (run_local_simulation(): build, health checks, simulate, analyze)
│   │   │   ├── run_collection.py        #   the shared steps of the collect scripts (collect_run(): health checks, collect, analyze)
│   │   │   ├── run_submission.py        #   the shared steps of the async submit scripts (submit_async_run(): config, build, layout, health checks, submit, submit.log)
│   │   │   ├── submit_with_retries.py  #   the retry loop of the EQE1 and VLQ *_with_retries.py submit scripts (run_with_retries(): waits for the latest job (VLQ; EQE1 optionally waits), relaunches the submit script)
│   │   │   └── venvs.py                 #   venv_python(): a venv's interpreter, on Windows and Linux
│   │   └── analysis/                    #   from records to results
│   │       ├── mitigation.py            #   expectation values, NEC/RC/cRC mitigation, AWAE, run_data*.json
│   │       ├── run_timing.py            #   CPU/QPU/queue time, failed attempts, QPU workload (prices) — see §10.2
│   │       ├── derived_quantities.py    #   summary quantities from AWAE and times, cumulative AWAE, EPSILON, best schemes; JSON helpers
│   │       └── collective_plots.py      #   plots comparing all analyzed runs (AWAE, QPU, submission and collection time, window, cost, N_acc, E_min) and their table as a CSV — see §10
│   ├── analysis_scripts/                #   every analysis script; run_post_process.py launches the others — see §10
│   │   ├── run_post_process.py          #   post-processing driver for a set of runs
│   │   ├── bacq_ps_qbcs_summary_json.py      #   per-run summary JSON
│   │   ├── plot_real_qpu_results.py     #   observable + max-error plots
│   │   ├── plot_awae.py                 #   AWAE plots
│   │   ├── plot_awae_nsamples_comparison.py  #   AWAE vs. number of samples
│   │   └── plot_health_checks.py        #   health check + QPU job time plot
│   ├── benchmark_scripts/               # the benchmark itself; every script takes the size on the command line (§4.3):
│   │   │                                #   mini: 8 samples, Nt = 1, 3, 200 shots · full: 480 samples, Nt = 1…19, 10 shots
│   │   ├── local_simulation_scripts/    # the benchmark on the local simulator — see §4.3
│   │   │   ├── benchmark_noiseless_lnn.py              # noiseless, LNN — builds the circuits for the LNN QPUs; AWAE should come out ≈ 0
│   │   │   ├── benchmark_noiseless_all_to_all.py       # noiseless, all-to-all — builds the circuits for AQT, VLQ and Sirius (QPY version 13)
│   │   │   ├── benchmark_coherent_noise_lnn.py         # LNN, with coherent CNOT noise compiled into the circuits
│   │   │   ├── benchmark_coherent_noise_all_to_all.py  # all-to-all, with coherent CNOT noise
│   │   │   └── benchmark_fake_fez_lnn.py               # LNN, with FakeFez noise (ibm.venv, offline)
│   │   └── real_qpu_scripts/            # one folder per real QPU — see §7
│   │       ├── IBM/            + readme_IBM.md
│   │       ├── IQM_EQE1/       + readme_iqm_eqe1.md
│   │       ├── IQM_RESONANCE/  + readme_iqm_resonance.md
│   │       ├── IQM_VLQ/        + readme_iqm_vlq.md
│   │       └── AQT_PIAST-Q/    + readme_aqt.md
│   │           └── PCSS_Quantum_UI/  # sbatch files for running the AQT benchmark on the PCSS cluster, and aqt_run_folder.py (prints a run folder; readme_aqt.md §4.3)
│   └── tests/                           # validation scripts (§9)
├── analysis/                            # analysis output only
│   ├── plots+summaries/                 #   per run and per collective set: the figures and a copy of the summary JSON(s); the collective run table (CSV)
│   └── tests/                           #   test output (git-ignored)
├── calibration_input_files/             # the latest provider calibration files (PIAST-Q, Euro-Q-Exa) for calibration_file= — see §5.4
├── datafiles/                           # all output: per-run folders, and the circuit caches in circuits/ (§4.1)
├── tokens/                              # the provider API tokens (git-ignored except README.md) — see §7
├── USER_MANUAL.md                       # this file
├── README.md                            # the public overview of the repository
├── GUIDE_FOR_ADDING_NEW_BACKENDS.md     # steps, records, tests and code changes for a new QPU provider — see §7
├── QUICK_START_Windows.md               # step-by-step path from a fresh clone to runs on every QPU (Windows)
├── QUICK_START_Linux.md                 # the same for Linux/macOS
├── LICENCE.txt                          # the EUPL v. 1.2 licence text
├── _images/                             # images used by the docs
└── venvs/                               # requirements_local_sim.txt + (git-ignored) virtual environments <name>.venv
```

---

## 3. Setup

### 3.1 Software modules

Python **3.12**. The base environment (local myQLM + local Qiskit/Aer simulation)
needs the packages below. Everything else in the environment is pulled in
transitively — including the whole `qat-*` / `myqlm-*` family, which `myqlm`
(a metapackage) hard-pins and installs automatically.

| Package | Version | Used for |
|---|---|---|
| `myqlm` | 1.13.7 | circuit build/compile + local simulation; **pulls** `myqlm-fermion` (BCS spin Hamiltonian, `qat.fermion`), `myqlm-simulators` (`get_default_qpu`), `qat-core/-lang/-comm/-devices`, … |
| `qat-qiskit` | 0.1.1 | myQLM → Qiskit circuit translation (`QaptivaToQiskit`) — *not* pulled by `myqlm` |
| `qiskit` | 2.5.2 | circuit representation and transpilation |
| `qiskit-aer` | 0.17.2 | local statevector / shot-based simulation |
| `numpy` | 2.4.4 | arrays throughout |
| `scipy` | 1.17.1 | `comb` for counting the number of two-body interaction terms |
| `matplotlib` | 3.10.9 | AWAE / observable plots |
| `tqdm` | 4.67.3 | progress bars |
| `tzdata` | 2026.5 | IANA time zone database for `zoneinfo`; installed on Windows only, which has none of its own |
| `pytest` | 9.1.1 | the test suite (§9) |
| `CairoSVG` | 2.9.0 | myQLM circuit SVG rendering |
| `pylatexenc` | 2.10 | Qiskit matplotlib circuit diagrams |
| `ipykernel` | 7.2.0 | *(optional)* running Jupyter notebooks in this venv |

> Real-QPU runs need a **separate** virtual environment per backend, IBM
> included. Those environments have **no myQLM**: they run on the circuits that
> `build()` caches as Qiskit QPY, so **run `build()` in this base environment
> first** (the `source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_*` scripts do it, §4.3). A real-QPU
> script whose circuits aren't cached yet stops with a message saying so.
>
> Each also needs its own provider packages — `qiskit-ibm-runtime` lives in `source/benchmark_scripts/real_qpu_scripts/IBM/requirements_ibm.txt`,
> **not** in this base `venvs/requirements_local_sim.txt`. For IQM VLQ and AQT the separate venv is
> also forced by their provider SDKs pinning incompatible Qiskit versions — e.g.
> `qiskit-aqt-provider` requires `qiskit < 2`, a full major version behind the
> `qiskit 2.5.2` pinned here. The exact modules and venv for each QPU are
> documented in that QPU's readme under `source/benchmark_scripts/real_qpu_scripts/<QPU>/` — refer to it
> before running on hardware.

### 3.2 Build the base virtual environment
#### Setup (Windows)
```powershell
py -3.12 -m venv venvs\local_sim.venv
.\venvs\local_sim.venv\Scripts\Activate.ps1
pip install -r venvs\requirements_local_sim.txt
```
(In PowerShell, allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`; `QUICK_START_Windows.md` §2 has the details.
Without activating, call `venvs/<name>.venv/Scripts/python.exe` directly.)

#### Setup (Linux)
```bash
python3.12 -m venv ./venvs/local_sim.venv
source ./venvs/local_sim.venv/bin/activate
pip install -r venvs/requirements_local_sim.txt
```
Without activating, call `venvs/<name>.venv/bin/python` directly.

Note: All virtual environments live inside the `venvs/` folder (git-ignored), named `<name>.venv`. Each backend gets its own virtual environment — `local_sim.venv` (base pipeline: local
simulator only, no provider SDK), `ibm.venv`, `aqt.venv`, ... and so on — each built from that QPU's own
`source/benchmark_scripts/real_qpu_scripts/<QPU>/requirements_<qpu>.txt`.

For instance, for the runs on IBM Quantum Platform, the virtual environment ibm.venv can be created by

Windows:
```powershell
py -3.12 -m venv venvs\ibm.venv
.\venvs\ibm.venv\Scripts\Activate.ps1
pip install -r source\benchmark_scripts\real_qpu_scripts\IBM\requirements_ibm.txt
```
Linux:
```bash
python3.12 -m venv ./venvs/ibm.venv
source ./venvs/ibm.venv/bin/activate
pip install -r source/benchmark_scripts/real_qpu_scripts/IBM/requirements_ibm.txt
```

The scripts meant to be launched — `source/benchmark_scripts/local_simulation_scripts/`, `source/benchmark_scripts/real_qpu_scripts/`
and `source/analysis_scripts/` (§10) — move to the project root
(`Quantum_Benchmarking_project/`) themselves, so they can be started from any directory — as can `pytest source/tests/` (the tests
do the same).

**Recommended**: right after setting up the base `local_sim.venv` above, and
before running anything else, verify the setup with the pytest suite —
run this from the project root, with `local_sim.venv` active:
```bash
pytest source/tests/
```
Runs `test_00`/`test_01`/`test_07`/`test_08`/`test_10`/`test_11`/`test_12`/`test_13`/`test_14`/`test_15`/`test_16`/`test_17`/`test_18`/`test_19`/`test_20`/`test_21`/`test_22`/`test_23`/`test_24`/`test_25`/`test_26`/`test_27` (§9) —
fundamental checks that confirm the circuits are built correctly, a full
local-sim pipeline + post-processing integration check, and unit tests of the
executed-circuit check and the timing bookkeeping — in 5 to 10 minutes (325 tests; longer on slower machines).
**Two venvs are needed**: `local_sim.venv` runs the suite (the run stops at once with instructions if `venvs/local_sim.venv` is missing; `test_10` also starts a child process in it), and `test_12` runs itself in `aqt.venv` — it fails with an error telling you to create `venvs/aqt.venv` (`source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/readme_aqt.md` §1, `requirements_aqt.txt`, which includes `pytest`) or to run `venvs/aqt.venv/Scripts/python -m pip install pytest` (`bin/python` on Linux) if only `pytest` is missing there.
All tests passing (no failures, no errors) means the base venv, myQLM↔Qiskit
translation, local Aer simulation, and `source/analysis_scripts/run_post_process.py` all work
before you go any further.

---

## 4. Quickstart — run the benchmark on the local simulator

All scripts below run from any directory (§3.2). Circuit-building times are from a desktop-class Windows PC; all times will differ on yours.
Treat them as rough guides.

**Suggested first runs.** To see how the benchmark works in about a minute, run the mini
benchmark (§4.3: 8 samples, Nt = 1, 3, 200 shots — the `mini` argument)
on the local simulator, in this order:

| # | Script (in [`source/benchmark_scripts/local_simulation_scripts/`](source/benchmark_scripts/local_simulation_scripts/)) | venv | What it shows | Approx. time |
|---|---|---|---|---|
| 1 | `benchmark_noiseless_all_to_all.py` | `local_sim.venv` | the whole pipeline end to end — build, health check, submit, analyze — on a noiseless simulator; AWAE is only shot noise | ~10 s including the circuit build |
| 2 | `benchmark_noiseless_lnn.py` | `local_sim.venv` | the same with linear-nearest-neighbor circuits (what the IBM, IQM EQE1 and IQM Resonance Emerald/Garnet runs use; IQM VLQ and Sirius run all-to-all) | ~10 s including the circuit build |
| 3 | `benchmark_fake_fez_lnn.py` | `ibm.venv` | the LNN circuits under realistic IBM Fez noise (§4.2), transpiled onto Fez's qubits; needs #2 first | ~35 s (no build of its own) |

A rerun of a finished run starts a new run folder (dated today, with the next run number); only
an unfinished run folder is resumed, skipping the stages already done (§5.4, §5.5).
Each script prints one line, `Build … s | simulation … s | analysis … s | total wall time … s`,
so you get the numbers for your own machine.

**Records and logs, as for a real QPU.** These scripts and the `qbcs_real_simulate_*.py` scripts
(§7) keep the same records as a real-QPU submission: the `[Run config]` block, a job record per
chunk in `job_records/` (§7), health checks before and after the simulation, and an entry in
`submit.log` (backend, chunks, shots, both health-check verdicts and, for the noisy ones, the
calibration and run qubits used). A crashed run leaves an `INTERRUPTED` entry there. A degraded
health check stops the noiseless and coherent-noise scripts (an `assert`: those simulators must
pass it); the scripts with a QPU's noise (`benchmark_fake_fez_lnn.py`, `qbcs_real_simulate_*.py`) warn
and continue, since the calibration they simulate may simply be poor.

**Seeing the circuits.** Every script that runs circuits — these, the real-QPU scripts (§7)
and every health check — saves the first circuit of every error-mitigation role (bare, NEC,
RC, RC+NEC, cRC, cRC+NEC), the first SPAM calibration circuit and the health-check circuit as
PDFs in `datafiles/<run folder>/circuit_drawings/`: `<role>_before_transpilation.pdf` and
`<role>_after_transpilation.pdf`, numbered per role (e.g. `01_bare_before_transpilation.pdf`;
`source/core/circuits/circuit_drawings.py`). The noiseless simulations run
the circuits as they are, so they have only the "before" drawings (their health check is still
transpiled). Nothing is drawn in the terminal.

### 4.1 Circuit caches and format

**Circuit caches.** The circuit cache (§5.4) is kept per topology *and* per
depth list: each depth list gets its own folder named by its Trotter steps, `datafiles/circuits/<topology>_Nt<steps>/`
(`_Nt1-3-5-7-9-11-13-15-17-19` for the full benchmark, `_Nt1-3` for the mini one; the coherent-noise
simulations add their noise, `_<noise type>_I<infidelity>_rct<ratio>`, so they never share a cache with the noiseless circuits),
because a chunk holds one circuit per (role, depth) and so can't be shared
between depth lists. Mini and full benchmarks therefore never touch each
other's circuits. The rest of what shapes the circuits — `sample_size`, `dt` and
`rng_seed` (a constructor parameter on `bacq_ps_qbcs`, default `0`, that seeds the
twirl generation) — is recorded in the folder's `cache_config.json`, and `build()`
stops with an error rather than reuse a cache built for another `dt` or `rng_seed`,
or for fewer samples than the run needs (a smaller run just uses the first chunks).
To rebuild with new settings, back up or delete the folder the error names and
rerun the script.

**Circuit format.** LNN circuits are cached at Qiskit's default QPY version
(`chunks_qiskit/chunk_{nc}_qiskit.qpy`); all-to-all circuits at QPY version 13
(`chunks_qiskit_v13_qpy/chunk_{nc}_qiskit_v13.qpy`), the newest the older Qiskit of
`aqt.venv` (1.4.6) and `iqm_vlq.venv` (2.1) can read — the two QPUs that run all-to-all
circuits. Newer Qiskit reads both, so a local all-to-all simulation runs exactly the files
sent to those QPUs. Loading these files in the older Qiskit would print a warning that
they were written by a newer Qiskit. Qiskit checks the writer's version recorded in the
header, not the QPY format version. Since this is the intended setup, `source/core/qbcs.py`
silences that one warning. A file that really can't be read still fails on load.

The full-size benchmarks and the arguments that choose the size are in §4.3.


**Progress output.** Build-time chatter (`Building jobs for X`, `Sending Job #0`,
per-chunk compile/interleave/conversion prints) is silenced by default and replaced
with `tqdm` progress bars for the genuinely long loops — circuit compilation/twirl
generation per depth, chunk interleaving, and myQLM→Qiskit conversion. Pass
`verbose=True` to `bacq_ps_qbcs(...)` to restore the line-by-line detail for
debugging.

Submission-time logging goes like the below:
```
[2026-09-15 23:59:55] Chunk 23: converting to Qiskit.
[2026-09-15 23:59:56] Chunk 23: submitting to backend.
[2026-09-15 23:59:57] Chunk 23: results received.
  [Chunk 23] Saved 30 count keys, 30 fidelity keys, SPAM counts for 8 input states.
[2026-09-15 23:59:57] qpu_chunk23_finished
```

**SPAM characterisation circuits.** When `do_SPAM=True`, `build()` also compiles
`2**nsites` SPAM characterisation circuits (8 for the 3-site model used here) — one
per computational basis input state. Every chunk's submission includes all 8 of
them alongside that chunk's own circuits (hence "SPAM counts for 8 input
states" above, on every chunk, not just once), so the calibration is measured under
the same run conditions as the data. They sit outside `sample_size`/`chunk_size` —
added on top, per chunk, not counted as RC samples.

**Timing in `run_data.json`.** `build_time_s`, `transpile_submit_time_s`, `qpu_runtime_s` and
`queue_time_s` (both including the failed jobs, §7; the full `timing` block) come from one breakdown, built from the labelled
intervals in the run's own `timestamps_utc.csv` (its `timestamps_local_time.csv` twin is for reading only), both appended to (never truncated)
across every `submit()` call on that run folder. A start pairs only with its own end and the
first complete pair of a chunk counts, so resuming a run does not double-count or give negative values.

**Circuit cache internals.** `datafiles/circuits/<topology>_Nt<steps>/compiled_from_plugins/`
holds the batch files, one per sample, depth and role (`batch_jobs_Nt<nt>_<role>_sample_<i>.batch`,
for `NEC`, `RC+NEC` and `cRC+NEC`, each holding two roles: sample indices `0…479` and `480…959`) — for
the default 480-sample, 10-depth run that's `2 × 480 × 10 × 3 = 28,800` files (a cache built for more samples holds more). The
interleaved circuits the jobs submit are in `chunks_qiskit/chunk_*_qiskit.qpy`. The
RNG seed is whatever `rng_seed` you pass to `bacq_ps_qbcs(...)` (default `0`, fixed
unless you change it) — see the cache-deletion note above for why the cache must be
cleared before changing it.

`build()` removes the empty folders (such as `None_circuit_twirls_*`) an internal debug-logging
path leaves in `compiled_from_plugins/` at the end of the build, deleting only folders that are actually empty.

**Disk space.** The full-benchmark circuit cache (`circuits/all_to_all_Nt1-3-…-19`, 10 depths ×
480 samples) is ≈3.1 GB for all-to-all and ≈4.1 GB for LNN (about 30,000 files each; a mini cache is ≈5 MB). A finished full run folder is ≈1–2 GB (IBM ≈1.9, IQM EQE1 ≈1.8, AQT ≈1.8, IQM VLQ ≈1.3, IQM Resonance ≈1.1), a mini run folder ≈2–3 MB. The local-simulator venv (`local_sim.venv`) is
reported at ≈1.5 GB.

### 4.2 Local simulator with a real backend's noise model

**Offline, with Qiskit's FakeFez.** `qiskit-ibm-runtime` ships frozen
calibration snapshots of IBM devices as fake backends (`FakeFez`, `FakeMarrakesh`,
`FakeTorino`, …).
[`source/benchmark_scripts/local_simulation_scripts/benchmark_fake_fez_lnn.py`](source/benchmark_scripts/local_simulation_scripts/benchmark_fake_fez_lnn.py)
(`mini`, ~35 s; or `full`, see §4.3) runs the LNN benchmark under `FakeFez`'s noise.
It makes no network calls and needs no token. It follows the same path as
`source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_simulate_ibm.py`: the fake backend's target is saved
to `datafiles/<run>/calibration_snapshot.json`, an Aer noise model is built from
it, the run's physical qubits are picked from it, and every chunk is transpiled
onto them. It needs **`venvs/ibm.venv`** (that's where `qiskit-ibm-runtime`
lives) and no myQLM, so the LNN circuits must already be cached — run
`benchmark_noiseless_lnn.py` (with the same benchmark arguments) in `local_sim.venv` first.
The snapshot is years older than today's device, so treat the results as
"a realistic superconducting QPU", not as a prediction for ibm_fez. For the current
calibration, use `qbcs_real_simulate_ibm.py` (§7).
The script takes `layout=optimized` (default), `random`, `lowest` or three qubits like `15,14,21` (§5.4), scored on
FakeFez's built-in calibration; `calibration_file=` is refused. The run folder ends in `layout-<mode>`.

**By hand, from a live backend.** Pull the calibration-derived `NoiseModel` from a real IBM backend and run it
through Aer locally. `layout_backend` must be passed so the circuits are
explicitly transpiled to that backend — without it, `AerSamplerV2` silently
cancels twirl gates.

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

service      = QiskitRuntimeService(channel="ibm_quantum_platform")
real_backend = service.backend("ibm_fez")
noise_model  = NoiseModel.from_backend(real_backend)

aer_backend  = AerSimulator(noise_model=noise_model)                 # for explicit transpilation
sampler      = AerSamplerV2(options=dict(backend_options=dict(noise_model=noise_model)))

qbcs.build(circuit_transpilation_LNN=True)     # LNN transpilation for a real-device topology

run_folder_path = qbcs.submit(
    run_folder     = "mini_sim-qpu-noise_ibm-fez_lnn-True_service-local_spam-True_layout-optimized_2026-10-11_#1",   # a run name (§5.4)
    lang           = "qiskit",
    service        = "local",
    backend        = sampler,
    layout_backend = aer_backend,              # <- required for the noisy path
)
qbcs.analyze_results()
```

Worked, end-to-end examples: the FakeFez scripts above and
`source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_simulate_ibm.py` (IBM noise model, 480 samples), and
[`source/tests/test_01_zero_noise.py`](source/tests/test_01_zero_noise.py)
(noiseless, Qiskit vs myQLM vs exact).

### 4.3 Minimal script sequence — build every circuit the QPU runs need

The real-QPU scripts (§7) never build circuits: their venvs have no myQLM, so
they load the circuits that `build()` cached in `local_sim.venv`. Run these
scripts once, in this order, before any real-hardware run — with the same
benchmark arguments as the real-QPU runs (below). Each builds a circuit set and
then runs the noiseless benchmark on it, which should give AWAE ≈ 0 (shot noise
only) — a check that the circuits are right before any QPU time is spent.

| # | Script (in [`source/benchmark_scripts/local_simulation_scripts/`](source/benchmark_scripts/local_simulation_scripts/)) | venv | Circuits it builds | Needed for | Full benchmark: building circuits / simulation once they're cached |
|---|---|---|---|---|---|
| 1 | `benchmark_noiseless_all_to_all.py` | `local_sim.venv` | `circuits/all_to_all_Nt<depths>/chunks_qiskit_v13_qpy/` (QPY version 13) | all-to-all runs: AQT PIAST-Q, IQM VLQ, IQM Resonance Sirius | ~60 min / ~15 min |
| 2 | `benchmark_noiseless_lnn.py` | `local_sim.venv` | `circuits/linear_nearest_neighbor_Nt<depths>/chunks_qiskit/` | LNN runs: IBM (Fez, Kingston, Marrakesh), IQM EQE1, IQM Resonance Emerald / Garnet | ~60 min / ~20 min |
| 3 | *(optional)* `benchmark_fake_fez_lnn.py` | `ibm.venv` | none — uses #2's circuits | a noisy reference under Qiskit's `FakeFez` noise (§4.2), offline | — (needs #2's circuits) / ~65 min |

(The simulation times were measured at 1000 shots per circuit.)

**Benchmark size — the command-line arguments.** Every benchmark script in `source/benchmark_scripts/` (submit, collect, job-status, simulate and the local
simulation scripts, also the real-QPU ones, §7) takes the benchmark size the same way
(`source/core/runner_support/benchmark_args.py`):

| Arguments | Benchmark | `<benchmark>` field of the run folder (§5.4) |
|---|---|---|
| `mini` | mini: 8 samples, Nt = 1, 3, 200 shots per circuit — cache `…_Nt1-3/` | `mini` |
| `full` | full: 480 samples, Nt = 1, 3, …, 19, 10 shots per circuit — cache `…_Nt1-3-5-7-9-11-13-15-17-19/` | `full` |
| `mini nshots_per_circuit=N` (or `full …`) | the preset, with N shots per circuit (same circuits) | `mini+nshots-N` (`full+nshots-N`) |
| `nshots_per_circuit=N n_samples=M n_depths=K` | custom: M samples, Nt = 1, 3, …, 2K−1 | `custom+ndepths-K+nsamples-M+nshots-N` |
| `nshots_per_circuit=N n_samples=M nt_list=1,5,9` | custom depth list (odd Nt) | `custom+nt-1-5-9+nsamples-M+nshots-N` |

A preset's samples and depths can't be changed, only its shots; `nshots_per_circuit`
without a preset needs both `n_samples` (a multiple of 8, so every 3-qubit readout bitstring
is covered equally often) and one of `n_depths` / `nt_list`. There is no default benchmark: a script
called without arguments prints its usage and does nothing, so name `mini` (it costs far less QPU time than the
full one, though not none) or `full`. For example:

```bash
python source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_lnn.py mini     # mini, a few seconds
python source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_lnn.py full     # full, ~60 min the first time
```

The local runs are saved in `datafiles/<benchmark>_<data type>_local-aer_lnn-<bool>_service-local_spam-<bool>_layout-<mode>_<YYYY-MM-DD>_#<N>/`,
with data type `sim-noiseless`, `sim-coherent` or `sim-fake-fez` — e.g.
`datafiles/mini_sim-noiseless_local-aer_lnn-True_service-local_spam-True_layout-none_2026-10-11_#1/` (§5.4; the
noiseless and coherent-noise runs have `layout-none`, the FakeFez run its layout, §4.2).

**Building the circuits** is the slow part, and it happens only once per topology
and depth list. For the full benchmark it takes about 60 min per topology (compiling the twirled circuits, interleaving them
into 480 chunks, and converting the chunks to Qiskit QPY; all-to-all: at QPY version 13, see §4.1).
The mini benchmark's take seconds (the noiseless LNN script: ~10 s including the
circuit build).
The per-stage times are logged in `datafiles/circuits/<topology>_Nt<steps>/timestamps_utc.csv`
(and, in local time for reading, `timestamps_local_time.csv`).

**Once the circuits are cached,** a rerun of the same script (into a fresh run
folder), or any other run of that topology and depth list, goes straight to the
simulation: loading the cached circuits takes seconds to a few minutes, and the
times in the last column are what's left. So the full minimal sequence #1 → #2 → #3
takes about 3¾ h the first time and about 1¾ h once the circuits exist.

A real-QPU script whose circuits aren't cached can't build them (no myQLM in its
venv), so it stops with a message pointing back to this step.

The circuit cache for the full-size runs (10 depths × 480 samples) is ≈3.1 GB (all-to-all) or ≈4.1 GB (LNN)
on disk per topology.

---

## 5. The pipeline — `bacq_ps_qbcs`

Three stages, always in this order:

| Call | Does |
|---|---|
| `qbcs.build(circuit_transpilation_LNN)` | build BCS Trotter circuits, compile the NEC/RC/cRC batches, interleave into chunks, cache to disk (§5.4) |
| `qbcs.submit(run_folder, lang, service, backend, …)` | dispatch every chunk to the target backend; for `service="local"` also merges results + builds the SPAM confusion matrix inline |
| `qbcs.analyze_results()` | compute mitigated expectation values / AWAE, write `run_data.json` (+ `run_data_spam.json`) — feeds §10 |

`qbcs._status` moves `"initialised"` → `"built"` → `"submitted"` → `"analysed"`.
`qbcs.reset_for_resubmit()` resets it to `"built"` so a fresh `submit()` (e.g. a
different shot count) can reuse the same built object.

Every stage writes its output to disk and skips itself if that output is already
there, so an interrupted run resumes by simply re-running the same call — see §5.5.

### 5.1 Constructor

```python
bacq_ps_qbcs(
    infidelity_total = 0.0,           # total injected gate infidelity, e.g. 5e-3 (0.0 = ideal; simulator studies only)
    ratio_ct = 0.0,                   # CNOT-to-total noise ratio, e.g. 0.5
    noise_type = "pauli",             # "pauli" | "depolarizing"
    nshots = 10,                      # shots per circuit (0 = exact statevector)
    sample_size = 480,                # number of RC samples (= chunks); a multiple of 2**nsites = 8
    chunk_size = None,                # circuits per chunk — n_depths * n_roles (e.g. 10 depths × 6 roles = 60);
                                      #   any other value is replaced by that, with a warning
    list_circuit_times = None,        # default np.arange(0.2, 4.0, 0.4) -> 10 depths
    infidelity_from_counts = True,    # NEC takes its infidelity from measurement counts
    nec_random_initial_state = True,  # NEC randomizes the input state
    do_SPAM = False,                  # must agree with "SPAM" in error_mitigation_strategies, if given
    SPAM_shots_per_state = 10,
    dt = 0.2,                         # Trotter step
    error_mitigation_strategies = None,   # default ["NEC","RC+NEC","cRC+NEC"], with "SPAM" first if do_SPAM
    rng_seed = 0,                     # seeds the twirl generation (recorded in the circuit cache, §4.1)
    verbose = False,                  # True: line-by-line build output instead of progress bars
    folder_prefix = "results",        # unused, kept for older scripts
    folder_path = None,               # unused, kept for older scripts
)
```

### 5.2 `build(circuit_transpilation_LNN=False)`

`False` → all-to-all connectivity. `True` → linear-nearest-neighbour transpilation
(use this to match a real device's 1-D chain layout).

### 5.3 `submit()` signature

```python
qbcs.submit(
    run_folder,                          # output subfolder name under datafiles/
    layout_connection = "all_to_all",    # or "linear_nearest_neighbor"
    lang              = "qiskit",
    service           = "local",          # "local" | "IBM_Cloud" | "MQSS_LRZ" | "IQM_Resonance" | "QAAS_IT4I" | "PCSS"
    backend           = None,             # AerSamplerV2(), an IQMBackend, a provider backend, …
    layout_backend    = None,             # AerSimulator(noise_model=…) for the noisy local path
    max_chunks        = None,             # async services only (IBM_Cloud, IQM_Resonance, MQSS_LRZ, QAAS_IT4I): submit only the first N chunks
    chunks_per_qpu_job_batch = None,      # >1 packs N chunks per hardware job (IBM_Cloud, QAAS_IT4I)
    max_circuits_per_job = None,          # IQM_Resonance, MQSS_LRZ: pack whole chunks into jobs of ≤ this many circuits (None: Resonance 100, EQE1 70)
    chunk_index       = None,             # PCSS: exactly one chunk, as this or chunks=[nc] (the AQT loop uses chunks=[nc])
    chunks            = None,             # list[int]: submit exactly these chunks (the submit scripts pass the chunks without results)
    physical_qubits   = None,             # list[int]: pin the run to these qubits instead of choosing them (§5.4)
)
```

For `service` values other than `"local"`, drive `submit()`/`collect_results()`
through the per-QPU scripts in §7 rather than calling them by hand.

`chunks` (and `chunk_index` for PCSS, which needs one of the two) exist because AQT is synchronous and blocking (§7): the AQT
submit script calls `submit(chunks=[nc])` once per chunk in its own external loop, one
chunk at a time, so a crash/reconnection only loses the in-flight
chunk rather than corrupting a longer batch. `max_chunks` keeps its original
meaning (an upper bound honored by a single call) for the async services.

### 5.4 Circuit cache, qubit layout and run names

`build()` caches the compiled circuits in a **topology-keyed shared folder**,
chosen by `circuit_transpilation_LNN`:

| `circuit_transpilation_LNN` | folder |
|---|---|
| `True`  (linear-nearest-neighbour) | `datafiles/circuits/linear_nearest_neighbor_Nt<steps>/` |
| `False` (all-to-all)               | `datafiles/circuits/all_to_all_Nt<steps>/` |

`<steps>` are the run's Trotter steps, e.g. `1-3-5-7-9-11-13-15-17-19` for the full benchmark's
10 depths and `1-3` for the mini one: a chunk holds one circuit per (role, depth)
and can't be shared across depth lists (`source/core/common.py`, `get_circuit_folder()`).

Each holds:

```
circuits/<topology>_Nt<steps>/
├── compiled_from_plugins/                    # per-depth / per-EMS / per-sample myQLM batches
│   └── batch_jobs_Nt{nt}_{ems}_sample_{ji}.batch
├── chunks_myqlm/
│   └── batch_jobs_interleaved_chunk{nc}.batch    # interleaved chunk batches (myQLM)
├── chunks_qiskit/                            # LNN caches
│   ├── chunk_{nc}_qiskit.qpy                     # raw (pre-transpilation) Qiskit circuits per chunk
│   └── spam_qiskit.qpy                           # raw SPAM calibration circuits (one per input state)
├── chunks_qiskit_v13_qpy/                    # all-to-all caches: the same, at QPY version 13, which the
│   ├── chunk_{nc}_qiskit_v13.qpy                 # older Qiskit of aqt.venv and iqm_vlq.venv can read (§4.1)
│   └── spam_qiskit_v13.qpy
├── cache_config.json                         # what shaped these circuits (rng_seed, dt, samples, circuit code version, ...); build() refuses a mismatch (§4.1)
├── exact_observables.json                    # noise-free values of the 7 observables per depth, which
│                                              # the analysis compares against (keyed by the Hamiltonian
│                                              # parameters, so a different g/ε/dt never reuses them)
├── timestamps_utc.csv                        # shared circuit-build stage timing, UTC — durations are computed from this
├── timestamps_local_time.csv                 # the same rows in machine-local time, for reading only
├── timestamps_qiskit_v13_qpy_utc.csv         # all-to-all caches: the build logs under these names (older
└── timestamps_qiskit_v13_qpy_local_time.csv  # caches may also hold the default-named ones)
```

A cache whose interleaved chunk files lie flat, directly under `circuits/<topology>_Nt<steps>/`
instead of in the `chunks_myqlm/` and `chunks_qiskit/` subfolders above, gets them copied (not moved)
into the matching subfolder the next time `build()` touches that cache; the flat copies stay in place
unless removed by hand.

These are keyed **only by qubit topology and depth list** — independent of
`infidelity_total`, `ratio_ct`, `nshots` and `sample_size` (a smaller run uses the first
chunks); `dt` and `rng_seed` are recorded in `cache_config.json` and must match (§4.1) — so every run
with the same topology and depth list reuses them. `build()` and the interleaving step are **per-file
idempotent: each file is written once on the first run and never overwritten**
(this generalises to the whole pipeline — see §5.5). To force a rebuild, delete
the folder.

Only the *backend-specific* transpiled (ISA) circuits are per-run — they live
under the run folder at `datafiles/<run_folder>/transpiled_circuits/`.

**Physical qubits.** Every chunk and the SPAM batch of a run are transpiled onto
the *same* physical qubits, chosen once per run and recorded in
`transpiled_circuits/physical_qubits.json` (`submit(..., physical_qubits=[…])`
overrides the choice). The choice maximises the estimated success probability
of chunk 0's deepest circuit from the backend's calibration data (2q-gate,
readout and 1q-gate errors) over all SWAP-free placements — see
`source/core/qpu/layout_selection.py`. The gate counts are those of the transpiled circuit: the
circuit is transpiled (locally) once onto any SWAP-free placement, and each virtual qubit's
native single-qubit gates are counted, virtual-Z `rz` excluded (the counts are the same on
every placement). IQM VLQ's star is scored per role instead — each virtual qubit's `prx`
gates, MOVE pairs and CZs with the resonator, over every ordered choice of qubits with
complete calibration (`source/core/qpu/iqm_vlq.py`, `readme_iqm_vlq.md` §4.1). Backends that report no error data
(IQM EQE1 via MQSS, AQT) are scored from a calibration file instead (below). Transpilation asserts that the 2q-gate count is unchanged and that all
circuits, SPAM included, sit on those qubits. Every health check is transpiled
like the chunks — for the backend, or for local runs for `layout_backend` (a
generic `AerSimulator` if none is given). Health checks run on the same
qubits once the run has recorded them (`submit_health_check(..., qubit_set=None)`
reads the record; an explicit `qubit_set` still takes precedence; the shared check in `submit_health_check()` exits with code 2 and one line before submitting if that set has a repeated or unusable qubit or a missing coupler; on IQM VLQ every set of distinct qubits 0…23 passes, since its target is all-to-all). Every
submit script therefore calls `resolve_run_layout()` (`source/core/qpu/calibration_input.py`, which wraps
`qbcs.resolve_physical_qubits(backend, run_folder=...)` and saves the calibration) right after `build()`, before
their first health check; a health check submitted
before any record exists gets qiskit's own layout. `source/analysis_scripts/run_post_process.py` warns about older runs whose
SPAM circuits sit on different qubits from their chunk circuits.

**The layout argument.** Every submit and simulate script takes `layout=` (`source/core/runner_support/benchmark_args.py`; the
usage of each script lists it):

| `layout=` | The run's physical qubits |
|---|---|
| `optimized` | the scorer above — the default for IBM, IQM Resonance and IQM VLQ. If the QPU has no SWAP-free placement the script exits with code 2 and one line, before any health check or job |
| `random` | one of all valid SWAP-free layouts on the QPU, drawn once with a fresh random seed; the seed is stored in `physical_qubits.json`, and a resumed run reuses the recorded layout |
| `lowest` | the lexicographically lowest valid layout |
| `15,14,21` | exactly these qubits (virtual qubit `i` → the `i`-th number; `[15,14,21]` also works, but brackets need quoting in most shells). A wrong number of qubits is rejected with the usage; if the QPU cannot run them (a qubit that does not exist, or a missing coupler) the script ends with a one-line `❌ Layout:` message and exit code 2 |

`calibration_file=PATH` goes with `layout=optimized`: the calibration to score the placements with. The file
may be in the project's snapshot format (`calibration_snapshot.json` of a run) or exactly as the provider
publishes it (Euro-Q-Exa's `calibration_data_…_active_qubits.json`, PIAST-Q's `piastq_characterisation_….json`);
a provider file is converted automatically (`source/core/qpu/calibration_input.py`) and saved in the run folder as
`calibration_snapshot.json`, so the run keeps the calibration it was scored with (a rerun with a different
file is an error). A relative `PATH` is relative to the project root. The latest provider files are in
`calibration_input_files/PIAST-Q/` and `calibration_input_files/Euro-Q-Exa/`.

**IBM, IQM Resonance and IQM VLQ report live calibration**, so their submit scripts refuse `calibration_file=` with
an explanation (exit code 2): a real run always scores the layout with the device as it is now, and saves that
calibration in the run folder. Their simulate scripts do take `calibration_file=`, e.g. to re-simulate an old run:
any snapshot for IBM and Resonance, and for VLQ a run's `vlq_calibration.json`, or that run's snapshot with its `vlq_calibration.json` beside it (its layout choice needs
IT4I's raw calibration). VLQ's resonator is always part of the layout, so `random`, `lowest` and a list choose the
three qubits that go with it. An explicit list always names exactly three qubits.

**AQT PIAST-Q and IQM EQE1 report no calibration data** through their SDKs, so a live run can only be
optimized with a calibration file. Without `layout=` and `calibration_file=` their submit scripts run on a
**random** layout and print a notice; `layout=optimized` without `calibration_file=` is an error. Their
simulate scripts take the calibration from `calibration_file=` too (default: the newest file in
`calibration_input_files/<provider>/`) and require `layout=` explicitly, because a real submission without a file is random
and the simulation must be told which behaviour to mimic. PIAST-Q is all-to-all with 20 qubits (ions 0…19, read
from the live backend), so any ordered triple of ions is valid; EQE1's valid layouts follow the qubits and couplers of
its live Target.

**The calibration a run worked with is saved in its run folder.** A real run records what its layout was chosen
on as `calibration_snapshot.json` (`source/core/qpu/calibration_input.py: save_live_calibration`, once per run — a resumed run
keeps the first): the calibration file when `calibration_file=` is given, otherwise the QPU's own current calibration
for IBM and IQM Resonance (VLQ: also `vlq_calibration.json`). For IBM and IQM Resonance the layout is scored on that
saved snapshot itself, so it is the calibration the layout was chosen on. The snapshot has two dates: `downloaded_at`
(when it was saved) and `measured_at` (when the provider made the calibration, UTC: IBM's last property update, the end of
the IQM calibration set, or a calibration file's own date; `null` when the provider does not say). Older runs were given
`measured_at` afterwards, with `measured_at_source` saying what it was taken from. AQT and EQE1 report none, so without a
file there is nothing to save. A layout that cannot be used (a forced list the QPU cannot run, no layout that both a file and the QPU
support) ends the script with a one-line `❌ Layout:` message and exit code 2 before anything is submitted.

The layout is part of the run folder name — `…_layout-optimized`, `…_layout-random`,
`…_layout-lowest` or `…_layout-15-14-21` (always three qubits) — so runs with different layouts of one benchmark
never collide. Collect and status scripts take `layout=` too; it is optional: without `layout=` and `date=` every run of that benchmark with something left to collect is collected (or reported) in turn, all layouts and dates; give `layout=` to pick one run (including an already finished one); `date=YYYY-MM-DD` alone takes each layout's latest run of that day, `date=YYYY-MM-DD_#N` each layout's run with that number (layouts without one are skipped; exit code 2 only when no layout has one). With no unfinished run the script exits with code 2 and lists the folders found.

**Run, plot and summary names** (`source/core/records/run_names.py`, the one place they are built and parsed):

```
<benchmark>_<data type>_<provider>-<backend>_lnn-<bool>_service-<svc>_spam-<bool>_layout-<mode>_<YYYY-MM-DD>_#<N>
```

Fields are separated by `_`; inside a field `-` joins a name and its value, and `+` joins the parameters of `<benchmark>`.

| Field | Values |
|---|---|
| `<benchmark>` | `full`, `mini`, `full+nshots-100` (a preset with other shots), `custom+ndepths-3+nsamples-8+nshots-50`, `custom+nt-1-5-9+nsamples-8+nshots-50`, `health-check+reps-1+nshots-100` (standalone health checks) |
| `<data type>` | `real-data`, `sim-noiseless`, `sim-coherent`, `sim-fake-fez`, `sim-qpu-noise` (a simulate script: local Aer with a QPU's calibration) |
| `<provider>-<backend>` | `ibm-fez`, `ibm-kingston`, `ibm-marrakesh`, `iqm-emerald`, `iqm-garnet`, `iqm-sirius`, `iqm-vlq`, `iqm-eqe1`, `aqt-piast-q`; `local-aer` for local simulations without a QPU's noise (also the `sim-fake-fez` runs, which carry FakeFez's noise); the simulated QPU's `<provider>-<backend>` for `sim-qpu-noise` runs |
| `lnn`, `spam` | `True`, `False`, or `none` (standalone health checks) |
| `service` | `ibm-cloud`, `iqm-resonance`, `qaas-it4i` (VLQ), `mqss-lrz` (EQE1), `pcss` (AQT), `local` (every simulation) |
| `layout` | `optimized`, `random`, `lowest`, three qubits (`15-14-21`), `legacy` (old runs: qiskit chose per circuit), `none` (no QPU run: the noiseless and coherent-noise local simulations and standalone health checks; `sim-fake-fez` runs carry their layout) |
| `<YYYY-MM-DD>_#<N>` | the UTC day the submit, simulate or health-check script first made the folder, and the number of the run that day (`#1` for the first) |

**The date tells runs of the same settings apart.** A submit, simulate or health-check script works in the latest
folder of its settings while that run is unfinished (not every chunk submitted, or simulated; for a standalone health
check, no result yet), whatever its date, so a rerun resumes it; once it is finished, the next run of the same settings
starts a new folder dated today (UTC) with the next run number of the day. A submit script's `--force-new-run` starts a new
folder even while the latest run of its settings is unfinished (that one is left as it is). With `layout=` or `date=` given, collect and status scripts read the
latest folder of that layout, or the one of `date=YYYY-MM-DD` (its latest run that day) or `date=YYYY-MM-DD_#N`; `date=` alone does this for every layout that has such a run. A folder name without a date is still read (as `date unknown`).

E.g. `full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized_2026-10-11_#1`. Figures name a run by its
QPU and kind of data, layout and date with run number (`ibm-fez — real QPU, layout optimized, 11/10/2026 #1`; in the plots comparing
several runs `ibm-fez — real QPU, lnn-True, layout optimized, 11/10/2026 #1`, ordered by QPU, layout and date). A run's plots go to
`analysis/plots+summaries/<run name>/`; the plots comparing several runs to `analysis/plots+summaries/<benchmark>_collective_<data types>/`
(e.g. `mini_collective_real-data+sim-noiseless`; with `run_post_process.py`'s filters the folder name ends in them, e.g.
`…_dates-2026-09-01-to-2026-09-30_backends-ibm-fez`). A run's summary is `processed_results/summary.json` in the run
(`summary+nsamples-N.json` for an analysis of its first N samples — short, since the folder already carries the name
and Windows limits paths to 260 characters), and `run_post_process.py` copies it, as
`summary_<run name>.json` (`summary_<benchmark>+nsamples-N_…json`), next to the run's plots in `analysis/plots+summaries/<run name>/`
and into every collective folder that shows the run. Every run folder follows this scheme.

### 5.5 Idempotency — the pipeline is a chain of checkpoints

A benchmark run is long and fragile: it compiles thousands of twirled circuits,
transpiles them for a specific backend, ships them to a QPU queue, and waits —
sometimes for days — for results. Any step can be cut short by a crash, a walltime
limit, a dropped connection, or a QPU going offline.

So the pipeline is built on one rule: **every stage writes its output to disk, and
every stage checks disk before doing any work.** Re-running a stage that already
finished costs a directory listing, not a recompile or a re-submission. You almost
never "start over" — you re-run the same script and it resumes where it stopped.

There are six checkpointed stages, in order (the table below lists their eight checkpoint artifacts, since stage 3 has two extra ones):

1. **Compile** the RC / cRC circuits from the myQLM plugins.
2. **Interleave** them into chunks — one slice across every EMS and Trotter depth.
3. **Convert** each chunk to Qiskit QPY — LNN circuits at Qiskit's default version,
   all-to-all circuits at version 13, which the older Qiskit of `aqt.venv` and
   `iqm_vlq.venv` can read (§4.1).
4. **Transpile** each chunk to a backend's native gate set.
5. **Submit** each chunk to the QPU.
6. **Collect** each chunk's results.

Stages **1–3 are shared**: they depend only on the qubit topology (all-to-all vs
linear chain) and the depth list, not on noise level, shot count, or sample size
(`dt` and `rng_seed` must match the cache, §4.1). They are written once into a
shared folder and reused by every run with the same topology and depths — so a
sweep over noise strengths or shot counts pays the circuit-building cost exactly once.

Stages **4–6 are per-run** and tied to one backend. A hardware run that dies
halfway is resumed by re-running the same `submit` / collect call — even from a
fresh Python process, because everything needed to continue lives in `job_records/`
and `raw_results/` on disk, not in memory.

To *deliberately* redo a stage, you delete its artifact (table below). Nothing is
invalidated automatically.

#### The checkpoints

| # | Stage | Produced by | Artifact on disk | Reused when it exists |
|---|---|---|---|---|
| 1 | Compile circuits from plugins | `build()` → `load_or_compile_batch()` | `<shared>/compiled_from_plugins/batch_jobs_Nt{nt}_{ems}_sample_{ji}.batch` | loaded instead of recompiling the RC/cRC twirls |
| 2 | Interleave into chunks (myQLM) | `build()` → `interleave_job_batches()` | `<shared>/chunks_myqlm/batch_jobs_interleaved_chunk{nc}.batch` | chunk skipped |
| 3 | Convert chunk to Qiskit QPY | `build()` → `convert_chunks_to_qiskit()` | `<shared>/chunks_qiskit/chunk_{nc}_qiskit.qpy` (LNN), or `<shared>/chunks_qiskit_v13_qpy/chunk_{nc}_qiskit_v13.qpy` (all-to-all) | chunk skipped |
| 3b | Save SPAM circuits as Qiskit QPY (SPAM runs only) | `build()` → `convert_spam_to_qiskit()` | `<shared>/chunks_qiskit/spam_qiskit.qpy`, or `<shared>/chunks_qiskit_v13_qpy/spam_qiskit_v13.qpy` | skipped; `submit()` loads the file instead of building SPAM in myQLM (falls back to myQLM, with a warning, if it's missing) |
| 3c | Compute the exact observables | `build()` → `cache_exact_observables()` | `<shared>/exact_observables.json` | depth skipped; the analysis reads the file instead of simulating the ideal circuit in myQLM (falls back to myQLM, with a warning, if a depth is missing) |
| 4 | Transpile chunk to backend ISA | `submit()` → `resolve_physical_qubits()`, then `transpile_chunk_for_backend_to_isa()` / `transpile_spam_for_backend_to_isa()` | `<run>/transpiled_circuits/physical_qubits.json`, `chunk_{nc}_transpiled.qpy` (+ `spam_transpiled.qpy`) | loaded instead of re-selecting / re-transpiling |
| 5 | Submit chunk to QPU | `submit()` → `submit_chunk_ibm()` / `submit_batch_ibm()` / `submit_batch_iqm()` / `submit_chunk_aqt()` | `<run>/job_records/job_record_chunk_{nc}.json` | chunk skipped if its `status` is `submitted` or `complete` |
| 6 | Collect chunk results | `collect_results(..., service_name=…)`; inline for `service="local"` | `chunk_{nc}` key in `<run>/raw_results/results_counts.json` + `results_fidelity.json` | chunk skipped if `status == complete` **and** both keys present (any service also skips a chunk marked `failed`) |

`<shared>` = `datafiles/circuits/<topology>_Nt<steps>/` (`all_to_all` or
`linear_nearest_neighbor`, chosen by `circuit_transpilation_LNN`; `<steps>` from the depth list)
`<run>` = `datafiles/<run_folder>/`

#### Shared vs per-run

- **Stages 1–3 are keyed only on qubit topology and depth list.** They do *not* depend on
  `infidelity_total`, `ratio_ct`, `noise_type`, `nshots`, or `sample_size` (a smaller run
  uses the first chunks; `dt` and `rng_seed` must match, §4.1), so every run with the same
  topology and depths reuses them — a noise or shot sweep builds circuits once (§5.4). Stages 1–2 are also shared *across* the
  LNN and all-to-all (QPY version 13) chunk formats (same myQLM artifacts either way); only stage 3
  is format-specific, and each format's own build/timing log
  (`timestamps_utc.csv` / `timestamps_qiskit_v13_qpy_utc.csv`, and their `_local_time`
  twins, §5.4) gets the rows of any stage the two share but this particular build
  call skipped appended from the other's (never its Qiskit-conversion rows), so both
  stay complete regardless of build order. Every timestamp log is append-only: nothing
  rewrites it. An attempt killed mid-job leaves an unmatched start, which the analysis
  skips when reading. The build time is the sum of its stages (myQLM compilation +
  interleaving, then this format's own conversion), so days between building the two
  formats are not counted.
- **Stages 4–6 are per-run and backend-specific.** Because every byte needed to
  resume is on disk, the collect step runs in a brand-new Python session: the
  real-QPU collect scripts rebuild `CONFIG` from the job records and are meant to
  be run that way.
- For `service="local"` there are no job records — stages 4–6 collapse, and
  idempotency is just the `chunk_{nc}` keys in `raw_results/`.

#### Forcing a stage to re-run

Remove the artifact, and for hardware also clear the chunk's status guard.

| Stage | To force chunk `nc` to re-run |
|---|---|
| **1 — compile** | delete the matching `compiled_from_plugins/batch_jobs_Nt*_*_sample_*.batch` (or the whole `datafiles/circuits/<topology>_Nt<steps>/` folder to rebuild everything). Affects **all** runs of that topology. |
| **2 — interleave** | delete `datafiles/circuits/<topology>_Nt<steps>/chunks_myqlm/batch_jobs_interleaved_chunk{nc}.batch` |
| **3 — convert to Qiskit** | delete `chunks_qiskit/chunk_{nc}_qiskit.qpy` (all-to-all: `chunks_qiskit_v13_qpy/chunk_{nc}_qiskit_v13.qpy`) in `datafiles/circuits/<topology>_Nt<steps>/` |
| **3b — SPAM circuits** | delete `chunks_qiskit/spam_qiskit.qpy` (all-to-all: `chunks_qiskit_v13_qpy/spam_qiskit_v13.qpy`) in `datafiles/circuits/<topology>_Nt<steps>/` |
| **3c — exact observables** | delete `datafiles/circuits/<topology>_Nt<steps>/exact_observables.json` |
| **4 — transpile** | delete `datafiles/<run_folder>/transpiled_circuits/chunk_{nc}_transpiled.qpy` (and `spam_transpiled.qpy` if the SPAM layout changed), **and** set `"status": "built"` in `job_records/job_record_chunk_{nc}.json`. Transpilation only runs inside submit, so this re-submits the chunk as well. |
| **5 — submit** | in `job_records/job_record_chunk_{nc}.json`, set `"status"` to anything other than `"submitted"` / `"complete"` (e.g. `"built"`), or delete the file. `submit()` then dispatches a fresh job for that chunk. |
| **6 — collect** | delete the `chunk_{nc}` entry from `raw_results/results_counts.json` and `results_fidelity.json` — or delete both files to re-collect every chunk (note `spam_calibration_counts_per_chunk.json` and `results_*_merged.json` also live in `raw_results/`). For a chunk stuck at `"failed"`, also reset its `"status"` to `"submitted"`. |
| **local run** | no job records — delete `raw_results/results_counts.json` + `results_fidelity.json` (+ `spam_calibration_counts_per_chunk.json`), or just the `chunk_{nc}` keys. |

Recognised `status` values in a job record are `submitted`, `complete`, `failed`.
Any other value — or a missing record file — makes `submit()` treat the chunk as
not yet submitted.

---

## 6. Configuration knobs that matter for a benchmark run

| Knob | Effect |
|---|---|
| `list_circuit_times`, `dt` | which Trotter depths `Nt` are swept (`Nt = round(T/dt)`) |
| `sample_size` | RC samples — statistical precision of the mitigated estimate |
| `nshots` | shots per circuit (`0` = exact statevector) |
| `error_mitigation_strategies` | which of `SPAM` / `NEC` / `RC+NEC` / `cRC+NEC` to run |
| `do_SPAM` | also produce SPAM-corrected outputs (`run_data_spam.json`) |
| `circuit_transpilation_LNN` (in `build`) | all-to-all vs 1-D-chain circuit layout |
| `infidelity_total`, `ratio_ct`, `noise_type` | inject a synthetic noise channel (simulator studies only) |
| `chunk_size` | set to `n_depths * n_roles` (any other value is replaced, with a warning; §5.1); controls job granularity |

---

## 7. Running on a real QPU

The hardware workflow is **script-based**, not an inline API call. The scripts are in
`source/benchmark_scripts/real_qpu_scripts/<QPU>/`, and each submit, collect, status and simulate script takes the benchmark size on the command
line, as in §4.3: `mini` runs the mini benchmark (8 chunks, Nt = 1, 3, 200 shots
per circuit), `full` the full one (480 chunks, Nt = 1…19, 10 shots per circuit), and
`nshots_per_circuit=… n_samples=… n_depths=…` a custom one. Every real-QPU script (submit, collect, simulate, status, health check) requires the machine as
`backend=NAME`, e.g. `backend=ibm_marrakesh` or `backend=emerald`; there is no default, and neither is there a default benchmark for the scripts that take one: it must be named too (the standalone `health_check_real_*.py` scripts take only `backend=`). Called without any argument (exit code 2) or with `--help` (exit code 0), every script — also the local simulation scripts and `source/analysis_scripts/run_post_process.py` — prints its usage, before any provider SDK is loaded, and does nothing; the usage has the same sections everywhere. The allowed names
are listed in the `backends.json` of each QPU folder, together with each machine's topology (`"lnn"`:
linear-nearest-neighbour or all-to-all circuits), which sets `LNN` for the run. A collect or status script takes the
same arguments as its submission, `backend=` included (`layout=` is optional: without `layout=` and `date=` every run of that benchmark with something left to collect is collected (or reported) in turn, all layouts and dates; give `layout=` to pick one run, including an already finished one; `date=YYYY-MM-DD` alone takes each layout's latest run of that day, `date=YYYY-MM-DD_#N` each layout's run with that number),
and with `layout=` or `date=` reads the latest run of that layout, or the one of `date=YYYY-MM-DD` or `date=YYYY-MM-DD_#N`. The
run folder names the benchmark, the QPU, the layout and the date (§5.4),
`datafiles/<benchmark>_real-data_<provider>-<backend>_lnn-<bool>_service-<svc>_spam-<bool>_layout-<mode>_<YYYY-MM-DD>_#<N>/` (e.g.
`mini_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized_2026-10-11_#1`, `mini+nshots-100_…_layout-15-14-21_2026-10-11_#1`), so runs of
different sizes, layouts and days stay apart. Rerunning a submit script continues the latest run of its settings while
it is unfinished (`--force-new-run` starts a new one regardless), else starts a new dated one. Analyze them with
`python source/analysis_scripts/run_post_process.py mini --include-real-data` (mini) or
`python source/analysis_scripts/run_post_process.py full --include-real-data` (see §10).

To add a new QPU provider, follow [GUIDE_FOR_ADDING_NEW_BACKENDS.md](GUIDE_FOR_ADDING_NEW_BACKENDS.md).

1. **PRE health check** — submit + collect a 3-qubit reference circuit; abort if the QPU is degraded. It is skipped, together with the submission, if all chunks are already submitted; on EQE1 it is also skipped (the submission goes ahead) when the run's latest job was submitted less than 21 min ago. **Retries and exit codes** (`require_functional_health_check`, `source/core/health_check.py`): a submission or collection failure that looks like a network/DNS blip is retried a bounded number of times — each attempt gets its own numbered record in `health_checks/` (every attempt stays on disk; nothing is overwritten or discarded), while a verdict that comes back degraded is never retried, since that's a real signal, not a blip. No chunk is ever submitted unless a PRE check comes back functional. If it still can't confirm the QPU is healthy, the script exits with no chunks submitted, using one of three distinct codes: `10` QPU degraded, `11` a check was submitted but its result couldn't be collected (health unknown), `12` a check couldn't even be submitted (health unknown). The POST health check (below) only retries its own submission — see there for why. The standalone `health_check_real_*.py` scripts use the same two verdict codes, `0` functional and `10` degraded (AQT: also `13` when its circuit is not executed within the time limit). The IBM and IQM standalone scripts also exit `2` for a problem you can fix (no argument, a missing or unknown `backend=`, a missing token, for VLQ a missing or incomplete `tokens/iqm_vlq_project_info`, and, where the script has a `QUBIT_SET` (not IBM), a bad one) and `1` for any other error (a provider or SDK error, a failed submission, collection or job); they never exit `11` or `12`, which come only from the accompanying checks of the submit scripts. AQT's health check is one synchronous call with no separate submit/collect step and its own rule (see `readme_aqt.md` §4): the standalone script exits `0` functional, `10` degraded, `12` not submitted or `13` not executed in time (`2` for an unusable argument or a missing token, `1` for any other error), and the loop script's PRE and POST checks exit `13` when the check's task timed out and `12` when it could not be submitted; a timed-out check is saved as a `failed` health check (its wait as queue time, no QPU time), otherwise nothing is written to disk on failure. AQT's loop script also exits with `3` (pause after `AQT_PAUSE_AFTER_HOURS`), `4` (a `STOP` file in the run folder), and its retrying sbatch file with `14` when it gives up (readme_aqt.md §4.3).
2. **Submit** — `qbcs.submit(..., service=<QPU>)`. Asynchronous for IBM and IQM (returns immediately). IQM Resonance and EQE1 jobs are packed: each job holds as many whole chunks, each with its own SPAM circuits, as fit under `max_circuits_per_job` (Resonance 100, EQE1 70) — for the full benchmark that is 1 chunk (68 circuits) per job, for the mini benchmark 5 chunks (Resonance) or 3 chunks (EQE1) of 20 circuits. IBM and IQM VLQ pack `IBM_CHUNKS_PER_JOB` = 48 / `VLQ_CHUNKS_PER_JOB` = 12 chunks per job (the submit scripts' `CHUNKS_PER_JOB`); synchronous for AQT (waits for each chunk and merges inline).
3. **Collect** — for IBM and IQM, a separate `qbcs_real_collect_*.py` retrieves results chunk-by-chunk and merges them (safe to re-run in a fresh session — see §5.5). AQT needs no collect step.
4. **Analyze** — `qbcs.analyze_results()`.
5. **POST health check** — confirm the QPU stayed healthy across the run. It is submitted right after the submission (for IBM/IQM it is collected by `qbcs_real_collect_*.py`; AQT does one after every chunk).

**Rehearse without QPU time.** Each backend folder also has a `qbcs_real_simulate_*.py`.
For IBM, IQM Resonance and IQM VLQ it downloads the device's calibration to
`datafiles/<benchmark>_sim-qpu-noise_<provider>-<backend>_…_layout-<mode>_<YYYY-MM-DD>_#<N>/calibration_snapshot.json` (for VLQ the download is `vlq_calibration.json`, from which the snapshot is built; a rerun warns and reuses it; remove
the run folder to download fresh calibration), builds an Aer noise model from
it, and runs the whole benchmark locally on the physical qubits a real submission would
pick (`source/core/qpu/qpu_snapshot.py`; details in each backend readme). Its data type `sim-qpu-noise` means
`source/analysis_scripts/run_post_process.py` treats it like any local run; its summary JSON names the
snapshot's backend and date. IQM EQE1 and AQT report no calibration data through
their SDKs, so their simulate scripts take a calibration file — `calibration_file=PATH`, the provider's file
(converted automatically) or a snapshot, by default the newest in `calibration_input_files/<provider>/` — and warn that it is not
backend data. They also require `layout=` (§5.4). Every simulate script takes `layout=` and an optional
`calibration_file=` (the noise data of the simulation instead of the download; for VLQ a run's `vlq_calibration.json`, or that run's `calibration_snapshot.json` with its `vlq_calibration.json` beside it).

**Which virtual environment does what.** Only the step that talks to a provider needs that
provider's venv; everything that reads saved files runs in `local_sim.venv`:

| Step | Script | venv |
|---|---|---|
| Build the circuits (once per topology and depth list; the full-benchmark caches already exist) | any `local_sim.venv` script that calls `build()`, e.g. `source/benchmark_scripts/local_simulation_scripts/` (with the same benchmark arguments) | `local_sim.venv` (myQLM) |
| Simulate a real backend locally: download its calibration (first run only), build the Aer noise model, run every chunk and health check | `source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_simulate_ibm.py`, `source/benchmark_scripts/real_qpu_scripts/IQM_RESONANCE/qbcs_real_simulate_iqm_resonance.py`, `source/benchmark_scripts/real_qpu_scripts/IQM_VLQ/qbcs_real_simulate_iqm_vlq.py` (the IBM and IQM Resonance scripts read the provider's token; VLQ's calibration API needs no token); `source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/qbcs_real_simulate_aqt.py` and `source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/qbcs_real_simulate_iqm_eqe1.py` read a provider file from `calibration_input_files/` instead and need `layout=` | the backend's venv (`ibm.venv`, `iqm_resonance.venv`, `iqm_vlq.venv`, `aqt.venv`, `iqm_eqe1.venv`) — it has the provider SDK for the download and `qiskit-aer` for the simulation, and reads the circuits from the QPY cache, so no myQLM is needed |
| Submit to / collect from the real QPU | `qbcs_real_submit_*.py`, `qbcs_real_collect_*.py` | the backend's venv |
| Post-process any run — real, legacy, converted or simulated | `source/analysis_scripts/run_post_process.py` (and the scripts in §10) | `local_sim.venv`. It needs no myQLM: the exact reference values come from the cache `build()` writes (`exact_observables.json` in the circuit folder) and are stored in `run_data*.json`, which the plots read; only a depth missing from that cache would be computed with myQLM |

A simulated run is a synchronous `service="local"` run: its summary reports QPU and queue time as 0
and all its time as CPU time. The calibration it used stays in the run folder
(`calibration_snapshot.json`). Rerunning the simulate script on an unfinished run reuses it — it prints a warning that it
uses pre-saved calibration data — and resumes the run; once the run is finished, a rerun starts a new dated run folder
with fresh calibration (to start over on an unfinished one, remove its folder `datafiles/<benchmark>_sim-qpu-noise_<provider>-<backend>_…/` first). To post-process it, run
`source/analysis_scripts/run_post_process.py` with `--include-simulations-real`: every layout of every QPU's `sim-qpu-noise_<provider>-<backend>_…` runs is already
covered by `SIMULATED_QPUS` in `RUN_THESE` (narrow it with `--layouts=<mode>`), or run
`python source/analysis_scripts/run_post_process.py nshots_per_circuit=<its shots> --include-simulations-real`.

Legacy runs (recorded without timing blocks) were converted to the current format, so they are
post-processed like any run; see below.

**Health check files.** Health checks that accompany a run (all real-QPU submit scripts and the
`source/benchmark_scripts/local_simulation_scripts/` local-simulator scripts, with `submit_health_check(..., accompanying=True)`) are saved
in the run folder, `datafiles/<run_folder>/health_checks/health_check_<N>.json`, with a single
auto-incremented counter. Each file holds `P011`, the `verdict` (`functional` if `P011 > 0.5`, else `degraded`), UTC
`submitted_at` / `upload_end_at` / `scheduled_at` / `finished_at` (`upload_end_at` is `"N/A"` for
synchronous backends such as AQT, which have no separate upload), `after_submission` (false: taken before the
submission, true: after it) and `chunks`: the chunks of the submission it accompanies. The submit
script decides that list once, before anything is submitted — the chunks without a `submitted` /
`complete` job record (`chunks_without_submission()`, or `bacq_ps_qbcs.chunks_to_submit()` for the
local simulator, which also skips chunks with complete results); it may have gaps, e.g. the chunks
of failed jobs being resubmitted — and passes the same list to `submit(chunks=...)` and to both
health checks (AQT: the PRE health check records the whole list, each POST check the one chunk it follows). A failed chunk record is not lost on resubmission: just before the new record is
written, it is copied to `job_records/failed_attempts/job_record_chunk_<nc>_failed_<k>.json`
(`k` = 1, 2, … per chunk; `source/core/records/job_records.py`). A failed AQT chunk also leaves a `failed`
record (the time of its finished batches and of the circuits finished inside the failing batch), an AQT PRE or POST health check whose task timed out a
`failed` health-check record (its wait as queue time and its transpilation, no QPU time), and collection saves what the provider still reports
for a failed job (billed QPU time, queue, start/end) in its `timing`. The analysis reports those
attempts under `failed_attempts` in `summary.json` (their QPU, queue and collection
time, and the totals including them; their transpilation and submission are already in the CPU
time; the failed jobs' circuits and shots are counted in the cost, §10.2), and the health check plot draws them as failed jobs `J<n>✗` (`J<n>✗s`: the submission
failed).

**How failed attempts are counted (the rules differ with how each provider bills).** AQT bills per circuit (each
circuit is its own PCSS task), so a failed attempt counts only the circuits and shots that finished before the
failure; an AQT health check that timed out adds nothing (PIAST-Q was unreachable). IBM, IQM Resonance, IQM EQE1 and
IQM VLQ work at job level, so a failed job that reached the provider (it has a `job_id`) counts in full: all its
chunks' circuits × `nshots`, and its billed QPU seconds (EQE1: modelled); a job shared by several chunks counts once. A submission
failure without a `job_id` is not counted. Submissions marked `may_exist_on_provider`, and jobs still `submitted` but
never collected (chunks and health checks), are not counted either; `cost_explanation` names them with an unknown
cost. The headline `QPU_time_s`, `Queue_time_s` and `Collect_time_s` (and so the `FT` fractions and the per-EMS
times) include the failed jobs' time, so they describe the whole run as it ran; the successful jobs' times
(`timing_breakdown`) and the failed jobs' (`failed_attempts`) stay beside them.

**Unattended submission (recommended).** The IQM EQE1 and IQM VLQ submit scripts have retrying twins, `qbcs_real_submit_iqm_eqe1_with_retries.py` and `qbcs_real_submit_iqm_vlq_with_retries.py` (same arguments). The VLQ one waits until the latest job submitted to the run has finished on its provider before every submission attempt, and gives up after 48 h without progress; the EQE1 one does not wait, retries a failed attempt at once and then every 1 min, gives up after 20 min without progress, and its submit script skips the health check before the submission when the run's latest job was submitted less than 21 min ago; when only the health check after the last chunks could not be submitted, they repeat just that health check, at most 3 times. Their readmes (§4) describe the exit codes.

**Chunk submission: retries and failed submissions (every backend).** On IBM and IQM (EQE1,
Resonance, VLQ) each chunk job's submission (`_timed_submit` in `source/core/qbcs.py`) sorts an
error into one of three kinds (`_submit_error_kind` in `source/core/qpu/remote_jobs.py`):
- *the provider was never reached* — DNS failure, connection refused, connect timeout: nothing was
  created, so the submission is retried, up to 3 more times after 30 / 60 / 120 s;
- *the answer was lost* — read timeout, connection reset or aborted, the VLQ upload's own time
  limit (`VLQTimeout`): the request may have arrived and the job may exist and run, so it is **not**
  retried (a retry could run the chunk twice);
- anything else (a rejected payload, a bug): not retried.

Every attempt is kept in the record as `submit_attempts`. When the submission gives up, each chunk
of the job gets a record with `status: "failed"`, `failure: "submission failed: <error>"`,
`failure_stage: "submission"`, `job_id: null`, a `submission_id` shared by the job's chunks (the
analysis counts the job once), `qpu_s: 0` and its transpilation and submission time; the error is
then re-raised, so the script stops (with its `submit.log` crash entry). A chunk of the
job that was already submitted keeps its record. **`may_exist_on_provider: true`** marks a lost
answer: check the provider for that job (its web interface; `check_real_qpu_job_status_<qpu>.py` in the IBM, IQM_EQE1, IQM_RESONANCE and IQM_VLQ folders, which
print the same report; LEXIS for VLQ) before rerunning the submit script, which resubmits every `failed` chunk. AQT keeps its
own per-circuit retry (`_aqt_retry` in `source/core/qpu/aqt.py`, connection errors only, within a 30 min limit per circuit, `AQTTaskTimeout`, which ends the script with exit code 13; pausing, the `STOP` file and automatic reruns on PCSS: readme_aqt.md §4.3); a chunk that still fails gets a
`failed` record with `failure_stage` `"submission"` (a circuit could not be submitted) or
`"execution"`. Collection marks provider-side failures with `failure_stage: "execution"`. Every
chunk record is written with `chunk_id`, `status` and the failure fields first. Health checks keep
their own retry rule (the PRE and POST checks' submission is retried on network errors only, up to 3 tries with 30 s, then 60 s waits, where a chunk submission gets 4 tries with 30 / 60 / 120 s) and record a failed submission (`status="failed"`,
`health_checks/`). Every submission that submits anything is followed by a health check, whatever
earlier checks covered. The analysis places a health check at the smallest chunk in its list
(which health checks belong to a partial analysis). Legacy records hold
`jobs` (one list of chunks per QPU job) and `prior_chunks` instead of `chunks`, and are still read. Local simulator runs
write the same chunk records as a real QPU (`service_name: "local"`, `status: "complete"` once the results
are saved, `submitted_at` / `finished_at`, a timing block without QPU or queue time, `circuit_check:
not_available` — the simulator returns no executed circuits); a simulation that raises leaves a `failed`
record (`failure_stage: "execution"`), archived when the chunk is simulated again. Legacy local runs have
`chunk_times_UTC_time.json` (per-chunk UTC start/finish) instead, which the analysis still
reads. A local run decides what to simulate by its results, not its records. The standalone
`health_check_real_*.py` scripts keep their single record in `job_records/health_check.json`.
Files named `job_records/health_check_<N>.json` (a layout the analysis does not read) are ignored.
A check whose submission itself failed (§7's exit code `12`, non-AQT backends only) is recorded
with `status="failed"`, `job_id=None`, `verdict=None` and a `failure` message instead of the usual
fields — the ordinary `number=None` collection (`qbcs_real_collect_*.py`) skips these, since there
is no job to poll. A check that was submitted but whose collection attempt failed keeps
`status="submitted"` and gains a `last_collect_error` field, so a later collection attempt (by the
same submit-script retry, or by `qbcs_real_collect_*.py`) still picks it up normally.
`source/analysis_scripts/plot_health_checks.py` counts and marks `"failed"` records separately from ones that are
merely still uncollected, and lists them in its JSON summary's `failed_health_checks`.

**Timing and circuit verification in the job records.** Every chunk record
(`job_records/job_record_chunk_<nc>.json`) and every health check record gets a `timing` block
(`source/core/records/job_timing.py`) and a `circuit_check` (`source/core/records/circuit_check.py`):

| | IBM | IQM Resonance | IQM VLQ | IQM EQE1 | AQT PIAST-Q |
|---|---|---|---|---|---|
| `transpile_s`, `submit_s` | measured | measured | measured (`submit_s` includes the upload) | measured | measured; `submit_s` = 0 (synchronous) |
| `qpu_s` | billed (`usage.qpu_charge_time_seconds`) | billed (raw `metrics.billed_seconds`) | IQM Resonance-equivalent billed time: ⌈compilation + execution⌉ from IQM's server timeline; `qpu_exec_s` = qaas' `remote_hw_runtime` | 0.9 s per circuit | per circuit, by shots: 3 s up to 10 shots, 15 s at 200 (the maximum), linear in between — 3 s for the 10-shot chunks, 8.68 s for the 100-shot health checks (`qpu_s_physics_model`: gate model) |
| `queue_s` | first execution span − created | `compilation_started` − `created` | `compilation_started` − end of upload (IT4I's HEAppE wait + IQM's queue) | scheduled − submitted (server, Europe/Berlin → UTC) | per batch: wall time − QPU time |
| `circuit_check` | `job.inputs["pubs"]` vs submitted | `GET jobs/{id}/payload` vs submitted | qaas `get_transpiled_circuits()` vs submitted | `not_available` | `not_available` |

The check compares the server's copy of the job with the submitted ISA circuits
(`transpiled_circuits/`) gate by gate — per qubit, the sequence of operations with their angles and
measured bits, plus circuit count and shots. A mismatch, or a check that could not run, is printed
as a warning and recorded; it never stops the collection. The analysis summarises the checks for
chunks and health checks separately (`circuit_verification` in `run_data*.json` and
`summary.json`) and prints a red warning line on every figure if any failed, naming
at most one chunk and one health check (`⚠ Circuit mismatch: chunk 12 and others; health check 15`).
EQE1/AQT records and simulated runs (`not_available`) are checked instead for all transpiled circuits — chunks, SPAM
batch, health checks — sitting on the same physical qubits (`qubit_consistency`; results cached in
`processed_results/circuit_qubit_check_cache.json`).

**Legacy runs (recorded without timing blocks and circuit checks).** Every such run in the repository
(the AQT full run and the runs in `datafiles/old_runs_full/`) was converted once to the current
format. A converted record carries `timing.converted_from_old_record: true` and
`upgraded_from_old_format_at`; the original files are kept in `<run>/job_records_before_upgrade/`.
IBM records got the billed QPU time and a circuit check from IBM. IQM Resonance did not keep the
old jobs, so those records were converted from their own fields (`timing.provider_unavailable` says
so). EQE1 and AQT records were converted from what they
hold, by the EQE1 and AQT rules above. An old AQT record that was not converted stops the analysis.


All health checks of a run are one circuit on the run's qubits: it is transpiled once, saved as
`transpiled_circuits/health_check_transpiled.qpy` (listed in `health_check_transpiled.json`) and that
copy is submitted every time. Its transpile time (the sum over the variants in
`health_check_transpiled.json`) is counted once in the run's CPU time; a subset analysis includes this whole-run time.

Each backend folder has a `qbcs_real_submit_*.py`, a standalone `health_check_real_*.py` and a
`readme_*.md`. IBM and IQM also have a separate `qbcs_real_collect_*.py` and a job-status script; AQT
runs synchronously, so its submit script runs the accompanying health checks and merges the results itself.

| QPU | Scripts folder | Submit / collect | Manual |
|---|---|---|---|
| IBM | `source/benchmark_scripts/real_qpu_scripts/IBM/` | async + separate collect | [`readme_IBM.md`](source/benchmark_scripts/real_qpu_scripts/IBM/readme_IBM.md) |
| IQM — EQE1 (via MQSS) | `source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/` | async + separate collect | [`readme_iqm_eqe1.md`](source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/readme_iqm_eqe1.md) |
| IQM — Resonance (cloud) | `source/benchmark_scripts/real_qpu_scripts/IQM_RESONANCE/` | async + separate collect | [`readme_iqm_resonance.md`](source/benchmark_scripts/real_qpu_scripts/IQM_RESONANCE/readme_iqm_resonance.md) |
| IQM — VLQ (IT4Innovations, via LEXIS/HEAppE) | `source/benchmark_scripts/real_qpu_scripts/IQM_VLQ/` | async + separate collect | [`readme_iqm_vlq.md`](source/benchmark_scripts/real_qpu_scripts/IQM_VLQ/readme_iqm_vlq.md) |
| AQT — PIAST-Q | `source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/` | synchronous, merged inline | [`readme_aqt.md`](source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/readme_aqt.md) |

**All-to-all QPUs (AQT, VLQ)**: `aqt.venv` (Qiskit 1.4) and `iqm_vlq.venv` (Qiskit 2.1) are pinned to
older Qiskit than the rest of the project and can't read the default QPY version. All-to-all
circuits are therefore always cached at QPY version 13 (§4.1):
`source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py` (§4.3) builds them in
`local_sim.venv` for both.

**Credentials** (`tokens/`): every backend reads a token from a plain-text
`tokens/<provider>_token` file (git-ignored except `tokens/README.md`), always through
`source/core/qpu/tokens.py: read_token()` (IQM VLQ locates its files with `token_path()`, see its rows), which exits with an explanation when the file is missing:

| Backend | Token file |
|---|---|
| IBM | `ibm_token` |
| IQM EQE1 | `iqm_eqe_token` |
| IQM Resonance | `iqm_resonance_token` |
| IQM VLQ | `iqm_vlq_token` — the LEXIS refresh token, written by the first (browser) login and rotated on every use (`readme_iqm_vlq.md` §2) |
| AQT | `aqt_piast-q_token` |
| IQM VLQ (access file) | `iqm_vlq_project_info` — the LEXIS project and resource (`tokens/README.md`) |

Every submit, collect, status and health-check script of that QPU (and the IBM and IQM Resonance simulate scripts) reads its token from this file
(through `source/core/qpu/tokens.py: read_token()`; IQM VLQ: `token_path()`).

Reading the token from its `tokens/<provider>_token` file and passing it explicitly into
the service/client constructor is how every QPU's submission script authenticates
— a deliberate feature of the scripts, not a workaround. No hidden or implicit
credential source is used anywhere.

---

## 8. Outputs

Every run writes to `datafiles/<run_folder>/`:

```
datafiles/<run_folder>/
├── raw_results/
│   ├── results_counts.json                    # per-chunk measurement counts (non-NEC roles)
│   ├── results_fidelity.json                  # per-chunk NEC-role counts
│   ├── spam_calibration_counts_per_chunk.json # SPAM calibration counts
│   └── results_*_merged.json                  # merged across all chunks
├── processed_results/
│   ├── run_data.json                          # mitigated observables, no SPAM correction (always)
│   ├── run_data_spam.json                     # with SPAM correction (only if do_SPAM=True)
│   ├── health_checks.json                     # all health checks and their jobs (health_checks+nsamples-N.json, run_data+nsamples-N.json,
│   │                                          #   summary+nsamples-N.json: the same for a subset analysis of N samples/chunks)
│   └── summary.json            # optional — written by source/analysis_scripts/bacq_ps_qbcs_summary_json.py, see §10
├── job_records/job_record_chunk_{nc}.json     # one per chunk: job id, status, timing block, circuit_check (§7) — local runs too
├── job_records/failed_attempts/               # records of submissions that failed and were retried
├── health_checks/                             # one numbered record per health-check attempt (PRE / POST)
├── circuit_drawings/                          # drawings of the run's circuits
├── run_info.json                              # what the run was: shots, samples, depths, dt, topology, SPAM (§10.2)
├── calibration_snapshot.json                  # real runs: the calibration the layout was chosen with (§5.4)
├── transpiled_circuits/                       # backend-specific ISA circuits (per-run): chunks, SPAM,
│                                              # the health check circuit, physical_qubits.json
├── timestamps_utc.csv                         # pipeline stage timing, UTC — durations are computed from this
├── timestamps_local_time.csv                  # the same rows in machine-local time, for reading only
└── submit.log                                 # human-readable summary, one entry per submission (every submit and
                                              # local simulation script; crashed attempts get an INTERRUPTED entry)
```

**Bitstring convention.** Every saved bitstring uses myQLM's ordering: the
*leftmost* character is qubit 0. Qiskit orders them the other way round (rightmost
= qubit 0), so the submit code reverses every bitstring it gets back from Qiskit
before saving it — benchmark counts and SPAM counts alike.

**Raw-results JSON schema.**

- `results_counts.json` (non-NEC roles) and `results_fidelity.json` (NEC roles) are keyed
  by chunk, then by job tag `job_Nt_<NN>_<role>_ems<i>`, then by circuit, then by bitstring:

  ```json
  {"chunk_0": {"job_Nt_01_raw_ems0": {"Nt_01_raw_circuit_0": {"000": 4, "001": 2}},
               "job_Nt_01_rc_ems2":  {"Nt_01_rc_circuit_0":  {"010": 3}}, "…": {}},
   "chunk_1": {"…": {}}}
  ```

  `results_fidelity.json` has the same shape with the NEC roles (`job_Nt_01_raw_nec_ems1`, …).
- `spam_calibration_counts_per_chunk.json`: chunk → prepared input state (`"000"` … `"111"`)
  → measured-bitstring counts, e.g. `{"chunk_0": {"001": {"001": 9, "011": 1}, …}}`.
- `results_*_merged.json`: after merging, a single `chunk_0` key whose job tags hold
  `circuit_0` … `circuit_<n_chunks-1>`, one per chunk (480 for the full benchmark).

Turning `run_data.json` into figures and a per-run summary deliverable is a
separate, manual step — see §10.

---

## 9. Validation / tests

`test_00`, `test_01`, and `test_07` are the fundamental tests — fast,
pytest-collected (`pytest source/tests/`, §3.2). `test_00`/`test_01` are zero-noise
checks that confirm the circuits themselves are built correctly (myQLM vs
exact, and Qiskit vs myQLM vs exact); `test_07` is a full end-to-end
integration check — builds, submits, and analyzes a local-sim run at two
shot counts, verifies every output file lands where §8 documents (including
the `summary.json` and all six observable/error/AWAE plot
PNGs from §10), runs `source/analysis_scripts/run_post_process.py`'s full post-processing
chain against it, and checks that AWAE is small and shrinks with more shots
(~1/√shots) for all six roles (bare, NEC, RC, RC+NEC, cRC, cRC+NEC) — a
ratio far below the expected scaling would indicate coherent residual noise
rather than ordinary shot noise. Run these first, right after setup.
The numbered tests skip 02-06 and 09 (those numbers belong to retired tests).

| Test | Checks |
|---|---|
| `source/tests/test_00_zero_noise_no_shot_noise.py` | myQLM pipeline vs exact statevector, infinite shots |
| `source/tests/test_01_zero_noise.py` | Qiskit vs myQLM vs exact, finite shots, zero gate noise |
| `source/tests/test_07_full_pipeline_local_sim.py` | full local-sim pipeline + `run_post_process.py` integration (outputs, summary JSON, plots), AWAE decay with shots for all 6 roles — no coherent residual noise; a second test runs the pipeline with `do_SPAM=False` (no SPAM files or figures) |
| `source/tests/test_08_mini_local_sim.py` | runs `source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py` with the `mini` argument (the mini benchmark) (output folder renamed), checks its circuit drawings and analyzes its results through `source/analysis_scripts/run_post_process.py` — summary/plots must carry its own 200 shots / 8 samples / 2 depths; run with and without SPAM (the no-SPAM variant replaces the script's `do_spam=True` argument with `do_spam=False`) |
| `source/tests/test_10_no_myqlm_on_qpu_path.py` | with myQLM imports blocked (as in the real-QPU venvs), `build()` from the cache, a local submission and the analysis all run without loading myQLM, and an uncached `build()` stops with the "build in local_sim.venv" message; every `source.core` module except `circuits/main.py`, `rc_functions.py`, `circuit_functions.py` imports with myQLM blocked |
| `source/tests/test_11_circuit_check_and_timing.py` | unit tests, no simulation: the executed-vs-submitted circuit check on real IBM/IQM Resonance job data (`source/tests/circuit_check_data_executed_vs_submitted/`, all must match except the fixture marked `mismatch`) and on altered synthetic circuits (all must mismatch; malformed server data gives `check_failed`, never an exception); the analysis summary incl. the qubit-consistency fallback; the figure warning line; Europe/Berlin → UTC incl. the repeated DST hour; durations from `timestamps_utc.csv` across a DST change, with fallback to the local-time log; append-only logs and skipping interrupted attempts; the build time as a sum of stages; refusing old AQT records without a timing block; reading old EQE1 runs, and records whose QPU time comes from their old fields, as legacy runs; `QPU_execution_time_s` per backend (summed over jobs, `N/A` when any job lacks it); the health check transpile time counted once per run (whole-run in a subset analysis); layout selection counting native single-qubit gates on the transpiled circuit; the shared UTC helpers (`parse_utc`, `iso_utc`, `now_utc_iso`) |
| `source/tests/test_12_aqt_shot_split.py` | **runs in `aqt.venv`, not `local_sim.venv`** (Qiskit 1.4.6, as AQT runs it): `test_12` only launches `source/tests/aqt_venv_test_12_shot_split.py` with `venvs/aqt.venv`'s Python, and fails with instructions if `aqt.venv` (or `pytest` in it) is missing. Unit tests, no simulation: AQT's 200-shots-per-circuit limit — the even split of larger shot counts (`job_timing.aqt_shot_parts`), `_aqt_run` refusing more than 200 shots, per-batch QPU-time modelling, and `submit_chunk_aqt` against a stub backend: parts checkpointed and summed per circuit, an interrupted chunk leaving a `failed` record (archived on the retry) and resumed at the first missing part, each execution counted as circuits in `qpu_workload`, and a failure inside a batch keeping the circuits finished before it in the `failed` record (circuits, shots, QPU time; its wall time booked once as queue time); a circuit not executed within `AQT_TASK_TIMEOUT_S` raising `AQTTaskTimeout` and leaving a `failed` record; the AQT connection-retry notices and a timed-out accompanying health check leaving a failed record; the shot parts' timestamps in `timestamps_*.csv`; the loop script's `STOP` file (honoured once, then deleted) and its `AQT_PAUSE_AFTER_HOURS` time limit, with their exit codes 4 and 3 |
| `source/tests/test_13_iqm_vlq.py` | unit tests, no simulation or network: VLQ's qubit selection on a saved calibration set (`source/tests/iqm_vlq_check_data/`): each virtual qubit's prx / MOVE / CZ counts read from a transpiled star circuit, every ordered choice of qubits with complete metrics scored by those roles, the effective all-to-all snapshot for the local simulation (written beside the raw calibration by `save_vlq_calibration()`), MOVEs not counted as two-qubit gates, and the timing of a real VLQ job (Resonance-equivalent billed time, `remote_hw_runtime`, IT4I/IQM queue split), and the connection log (every call to IT4I / HEAppE logged and time-limited, a hung request reported with where it waited), and the simulate script's `calibration_file=` (a run's `vlq_calibration.json`, or a snapshot accepted only with its `vlq_calibration.json` beside it) |
| `source/tests/test_14_incomplete_runs_and_collection.py` | unit tests, no simulation or network: runs with uncollected chunks in the middle — merging keeps each chunk's own number (the NEC input state depends on it), an analysis of n chunks covers the first n collected ones (timing, workload), a subset analysis corrects SPAM with every collected chunk's calibration (no `+nsamples-N` SPAM matrix by default), the `completeness` block and its warning; the collection time (`chunk{nc}_collect_*` labels, health checks' `collect_s`, N/A for older runs, `FT_collect`); non-blocking collection of chunk jobs and health checks (finished ones collected in any order, unfinished left pending, failed ones marked failed) against a faked provider; the chunks a submission covers (no `submitted` / `complete` record, gaps included), required by every accompanying health check as `chunks` (older `jobs` / `prior_chunks` records still read); the health check plot of an incomplete run (`+nsamples-N` files, uncollected checks and jobs); collecting a run while no job has finished analyzes nothing and does not fail, and collecting an aborted run (no chunk submitted) ends with one `❌` line (exit code 2); one reading of provider job-status texts (`job_state_of_status()`: a cancelled job counts as failed) shared by collection and the status report |
| `source/tests/test_15_health_check_network_retries.py` | unit tests, no simulation or network (a fake backend/collect-service stands in for the provider): `_is_transient_network_error`'s classification of connection/DNS/timeout errors (incl. walking `__cause__`); `submit_health_check`'s async-service (`IBM_Cloud`/`MQSS_LRZ`/`IQM_Resonance`/`QAAS_IT4I`) submission-failure handling — writes a numbered `status="failed"` record with its transpile and submit timing and raises `HealthCheckSubmitError` — vs. AQT/local and the standalone path, which stay unwrapped (raw exception propagates, nothing written); `require_functional_health_check`'s bounded retry on transient errors only and its three distinct exit codes (`EXIT_HC_DEGRADED` / `EXIT_HC_NOT_COLLECTED` / `EXIT_HC_NOT_SUBMITTED`) — a degraded verdict is never retried, and a failed collection attempt submits a fresh check under a new number rather than re-polling the same job; `submit_post_health_check` never raising or exiting (chunks are already on the QPU by then); and `collect_health_check(number=None)` / `plot_health_checks.py` handling a run with a mix of failed, uncollected and complete health checks without error, with failed ones counted/labelled separately; and the one verdict rule for every QPU (`health_check_verdict`: P(011) above the threshold), the shared verdict lines, P011 + the verdict in a synchronous standalone record, and an async standalone check keeping its collected result (counts, P011, verdict) in its own record |
| `source/tests/test_16_failed_job_attempts.py` | unit tests, no simulation or network (a temporary run folder): a failed chunk record archived to `job_records/failed_attempts/job_record_chunk_<nc>_failed_<k>.json` before a resubmission overwrites it (`k` counts up, the same failure never archived twice, other statuses left alone); readers of `job_record_chunk_<nc>.json` never taking an archived file for a chunk; `chunks_without_submission()` still listing failed and missing chunks; `failed_attempts_breakdown()` summing the failed jobs' QPU / queue / collection time (a batched job shared by several chunks counted once, each AQT attempt a job of its own, its executed circuits and shots summed and reported by `qpu_workload` as `n_failed_circuits` / `n_failed_shots`; IBM-style failed jobs shared by several chunks counted in full once, with string fields, plus a submission failure with `may_exist_on_provider` and an uncollected job reported as `n_may_exist_on_provider` / `n_uncollected`); the health check plot's jobs marking failed ones; chunk submission against fake IBM / IQM samplers — `_submit_error_kind` (provider never reached → retried; answer lost, the job may exist → not retried; other errors), retried submissions keeping `submit_attempts`, and a submission that gives up writing `"submission failed"` records for every chunk of the job (one `submission_id`, `may_exist_on_provider`, chunks already submitted left alone) and re-raising, archived on the rerun; records written with `status` near the top; the run window (`_job_intervals_for_window()`) spanning failed attempts and health checks, and its active length (`_active_seconds()`) without dead time (gaps between merged jobs over 30 min, 2 h on AQT, subtracted) |
| `source/tests/test_17_local_job_records.py` | unit tests, Aer only (no myQLM build, no network, a temporary simulation run folder): a locally simulated chunk writes a real-QPU-style `job_records/job_record_chunk_<nc>.json` (`status: "complete"`, `service_name: "local"`, times, timing block, `circuit_check: not_available`, `CONFIG` without `folder_path`, no absolute path); a sampler error writes a `failed` record (`failure_stage: "execution"`) and re-raises, and the rerun archives it to `failed_attempts/` where the analysis counts it; `chunks_to_submit()` of a simulation run decides by its results; the health check plot's jobs from new records and old `chunk_times_UTC_time.json`, also mixed; the shared `submit.log` entry (`append_submit_log`: layout, run qubits and calibration in a fixed order), `health_check_log_line`, `print_run_config`; `snapshot_log_lines`' calibration summary |
| `source/tests/test_18_layout_args.py` | unit tests, no simulation or network: the `layout=` / `calibration_file=` arguments and their errors (refused by the noiseless and coherent-noise local simulation scripts; the FakeFez one takes `layout=` but not `calibration_file=`), the four layout modes (optimized, random with a recorded seed, lowest, a given list and the errors for an impossible one) on synthetic targets incl. AQT's coupling-free one, the calibration formats (provider files vs snapshot) with automatic conversion, the date a snapshot records for the provider's calibration (`measured_at`) and the layout scored on the saved snapshot, and the run-folder names (the latest dated folder of a collect script, or that of `date=`, which without `layout=` takes each layout's run) |
| `source/tests/test_19_circuit_cache_and_venvs.py` | unit tests, no build or network: the circuit caches live in `datafiles/circuits/` (`datafiles/tests/circuits/` for the tests, which add nothing elsewhere), a cache built by another or no `CIRCUIT_CODE_VERSION` is refused, and `venv_python_path()` finds the interpreter on Windows and Linux; a missing token or LEXIS project file ends the script with one ❌ line and exit code 2 (`exit_with_error()`); `scrub_paths()` takes the project and home folders out of saved error texts |
| `source/tests/test_20_run_names.py` | unit tests, no simulation or network: run, plot and summary names (`source/core/records/run_names.py`) round-trip for every QPU folder and data type, the `<benchmark>` field of every kind of benchmark arguments, subset/summary/collective names, the service name of every QPU and of each run-name service field (`service_name()`); the figure label of every data type (`figure_label`, simulations marked as such); the run filters of `run_post_process.py` (`RunFilter`, `make_run_filter()`, `parse_date_filter()`) and the collective folder name of a filtered set; the date field (dated and undated names, `date_label`, `figure_label_full`, `dated_run_folder()` reusing only an unfinished run, `latest_run_folder()`) and the collective plots' run labels and order and time units (`_duration_fmt`: s, min, h, d), the run table CSV (`_write_run_table`, written by `make_collective_plots` as `<collective folder>/<collective folder name>.csv`) and the linear bar plots (`_bar_plot`); the data type of a `RUN_THESE` entry (`entry_data_type()`); the coherent noise of a run read from its job record (`coherent_noise_of_run()`) and the circuit cache folder it names; malformed names (and layouts of other than three qubits) refused |
| `source/tests/test_21_docs_consistency.py` | unit tests, no simulation or network: the user docs (README.md, this manual, both QUICK_STARTs, the guide for new backends, `tokens/README.md`, the QPU readmes) have no TAB bytes, every `source/…` path they name exists, every relative link and in-page anchor resolves, no user doc names the maintainer-only folder, every `USER_MANUAL §N` reference (in the docs and the requirements files) names an existing section and none points to a `README §N`, every token file is named `tokens/<provider>_token` (never the reversed order), every QPU readme has the shared seven top-level sections (Prerequisites, Connecting, Config knobs, Workflow, Timing, billing and circuit check, Outputs & post-processing, Known issues) and the shared paragraphs in the same words, and every `readme_<qpu>.md §N` reference exists |
| `source/tests/test_22_output_format.py` | unit tests, no simulation or network: the shared output helpers — `section()` headers, `print_done_block()` closing blocks, the one-line `print_chunk_saved()`, the two-line `format_retry()` notice (exception cut to 80 characters) and `top_counts()` for health-check counts |
| `source/tests/test_23_no_absolute_paths_in_saved_files.py` | runs the mini local simulation and its post-processing in a test folder, then checks that no text file of the run folder or of its `analysis/tests/` output (job records, `run_info.json`, `submit.log`, health checks, `run_data*.json`, summaries) contains an absolute path (the project root or home folder, a drive-letter or UNC path, `/home/`, `/Users/`, ...); the patterns are checked against example paths and URLs too, and `saved_config()` is checked to keep JSON-writable values (like the `coherent_noise` dict) and drop arrays and machine-specific paths |
| `source/tests/test_24_ibm_batch_and_optimized_layout.py` | `ibm_batch()` yields a sampler bound to the batch and closes it on success, on an error and with the default `max_time`; `layout=optimized` on a QPU with no SWAP-free placement exits with code 2 and one line without writing `physical_qubits.json`; an optimized or explicit layout that has a placement still resolves; the VLQ calibration source without an absolute path; `run_local_simulation.py` exiting cleanly |
| `source/tests/test_25_health_check_qubit_set.py` | an explicit health-check `qubit_set` is checked against the backend before anything is transpiled: a coupled set passes; an uncoupled, repeated, unusable or wrong-length set exits with code 2 and one line; no `qubit_set` and a local run without a layout backend are not checked; a VLQ-like all-to-all target accepts any distinct in-range qubits; the message names the backend the script passes (`backend_name`), else the backend object's own name |
| `source/tests/test_26_awae_helpers.py` | `cumulative_awae()` (the Z-weighted cumulative AWAE) equals the form the summary uses bit for bit and agrees with the other forms of it; NaN propagates; `EPSILON` is 0.05 and the default of `compute_derived_quantities`; `best_schemes()` lists the best scheme at $N_{max}$ and the one at $N_{acc}$ when different, skipping a missing one |
| `source/tests/test_27_submit_with_retries.py` | the retry loop of the EQE1 and VLQ `*_with_retries.py` submit scripts (`submit_with_retries.py`) with fake status and launch: the latest job (chunk or health check), the final statuses that launch and every other status that waits, the 10 min steps, the 48 h budget and its restart on progress, a folder that never appears, the exit codes, `--force-new-run` only until the folder exists, progress on a new run, and the final health check (repeated 3 times with `--retry-health-check`, never a new run; a failed check before the submission keeps the normal wait), and the EQE1 setting (no job wait, the first retry at once, 1 min waits, a 20 min budget) with the 21 min age below which the check before the submission is skipped |

### 9.1 Your circuit caches are safe during `pytest source/tests/`

The tests build their circuit caches only in `datafiles/tests/circuits/`: `source/tests/conftest.py`
points every cache path there for the session (`test_00`/`test_01` clear them repeatedly to start each
check from a clean slate, and the folder is emptied when a session starts). The caches your runs use,
`datafiles/circuits/<topology>_Nt<steps>/`, are never read, moved or deleted by the tests, so pytest can
run while a benchmark script is running, and nothing needs restoring after an interrupted test run.

Everything else the tests write stays in test folders too: their run folders in `datafiles/tests/`
(`tests/test<NN>…`), and their plots and summaries in `analysis/tests/plots+summaries/`.
Both `datafiles/tests/` and `analysis/tests/` are git-ignored, and your own runs, plots and summaries are
never touched.

---

## 10. Post-processing — plots and the summary deliverable

Once `qbcs.analyze_results()` (§5) has written `run_data.json` (and
`run_data_spam.json`, if `do_SPAM=True`), five more scripts turn that into
figures and a per-run summary. They take no command-line arguments.
`source/analysis_scripts/run_post_process.py` drives them through environment variables
(`QPU_RUN_FOLDER`, `QPU_N_SAMPLES`). To run one on its own, set `RUN_FOLDER` (the run folder's name,
§5.4) at the top of the file, and optionally `N_SAMPLES` (`None` = the whole run; an integer = the subset
of that many samples), then run it as a plain script. Everything else — the QPU, the topology, whether it
is a simulation — comes from the folder's name and its `run_info.json`.

The usual way to run all of them is `source/analysis_scripts/run_post_process.py`, which runs merge +
analysis and every script below for a set of runs, then draws plots comparing all of those
runs (`source/core/analysis/collective_plots.py`, see below). Without any argument, or with `--help`, it prints its usage and does nothing. Its first argument selects that set (required, no default), and at least one
`--include-*` flag selects the kinds of runs in it to analyze:

| Command | Analyzes |
|---|---|
| `python source/analysis_scripts/run_post_process.py mini --include-real-data` | the mini benchmark: the runs `mini_<entry>…` of `RUN_THESE["mini"]` — of the kinds the `--include-*` flags select |
| `python source/analysis_scripts/run_post_process.py full --include-real-data` | the full benchmark: the runs `full_<entry>…` of `RUN_THESE["full"]` — of the kinds the `--include-*` flags select |
| `python source/analysis_scripts/run_post_process.py <benchmark arguments> --include-real-data` | any other arguments of the `source/benchmark_scripts/` (§4.3; not `layout=` or `calibration_file=`), e.g. `mini nshots_per_circuit=100` or `nshots_per_circuit=50 n_samples=8 n_depths=3`: every run folder in `datafiles/` with that `<benchmark>` field (`mini+nshots-100`, `custom+ndepths-3+nsamples-8+nshots-50`) |
| `python source/analysis_scripts/run_post_process.py nshots_per_circuit=N --include-real-data` | every run folder in `datafiles/` with N shots per circuit (read from its `run_info.json`), whatever its size |

For `mini` and `full`, an entry that matches several run folders (e.g. one per layout) analyzes all of them, and an entry with no
run folder is skipped silently. Only the entries of the selected kinds (the `--include-*` flags below) are looked up; the script lists
the run folders it analyzes, one per line, before starting.

| Include flag (combine freely; **with none, nothing is analyzed** — the usage is printed) | Adds |
|---|---|
| `--include-real-data` | real-QPU runs (data type `real-data`) |
| `--include-simulations-real` | simulations with a real backend's noise: `sim-qpu-noise` (the `qbcs_real_simulate_*.py` scripts) and `sim-fake-fez` |
| `--include-simulations-noiseless` | noiseless simulations (`sim-noiseless`) |
| `--include-simulations-coherent` | simulations with coherent CNOT noise (`sim-coherent`, `benchmark_coherent_noise_*.py`) |

Either scope flag can be added as well (e.g. `python source/analysis_scripts/run_post_process.py mini --include-real-data --collective-only`):

| Flag | Does |
|---|---|
| *(none)* | both: the per-run analysis and plots, then the collective plots |
| `--individual-only` | only the per-run analysis and plots |
| `--collective-only` | only the collective plots, from the runs' existing results (nothing is re-analyzed; a run without results is left out) |

`--latest-per-qpu` keeps only the latest run (by date, then run number; complete or not, any layout) of each
`<provider>-<backend>` among the runs the selection and filters leave. Its collective plots go to a folder ending
`_latest-per-qpu`. In the collective plots and the run table, a run analyzed on fewer chunks than its sample size
(e.g. 300 of 480) is marked `(300/480 chunks)`. Every collective plots folder also holds a copy of the summary json of each run it shows.

`--nsamples-comparison` (full benchmark only, e.g. `python source/analysis_scripts/run_post_process.py full --include-real-data --nsamples-comparison`)
additionally analyzes each run on its first 8, 48, 120, 240 and 480 samples (`NSAMPLES_COMPARISON`; `run_data*+nsamples-<N>.json`
and `summary+nsamples-<N>.json` only, no per-N plots; only the RC samples vary, the SPAM correction uses the calibration of every collected chunk) and draws `analysis/plots+summaries/<run folder>/awae_best_ems_nsamples_comparison.png`
(`source/analysis_scripts/plot_awae_nsamples_comparison.py`). It shows the AWAE of the run's best EMS at N_max (and at N_acc, if that one differs)
for each sample count, with the run's own N_acc. Both come from the run's reference analysis: all its collected chunks, a curve that is always drawn.
Sample counts above that are left out. This plot is per run only: the collective plots are made as usual, from the full runs, and with `--collective-only` the flag is ignored.

**Filters** narrow the selected runs further, for the per-run and the collective analysis alike. All are optional and combine
freely (a run must pass every filter given; within one list any value matches); runs are matched on the fields of their folder
name (§5.4):

| Filter | Keeps the runs whose … |
|---|---|
| `--date=2026-09-01` | date is that day |
| `--date=2026-09-01:2026-09-30` | date is in that range, ends included; `--date=2026-09-01:` and `--date=:2026-09-30` leave one end open (a name without a date never matches) |
| `--backends=ibm-fez,iqm-garnet` | `<provider>-<backend>` is listed |
| `--services=ibm-cloud,pcss` | service is listed (`local` for the local simulations) |
| `--layouts=optimized,random` | qubit layout is listed (`optimized`, `random`, `lowest`, `legacy`, `none`, or three qubits such as `15-14-21`) |

E.g. `python source/analysis_scripts/run_post_process.py mini --include-real-data --date=2026-09-01:2026-09-30 --layouts=optimized`.
The script prints the active filters before the run list, and does nothing if no run matches. The collective plots and table of a
filtered set go to their own folder and file, named like the unfiltered ones plus the filters
(`mini_collective_real-data_dates-2026-09-01-to-2026-09-30_layouts-optimized/`; a long list of values is shortened to its count
and a hash), so they never overwrite the unfiltered comparison. A date, service or layout that is not valid, or a backend not written as `<provider>-<backend>` (e.g. `ibm-fez`), prints the usage and exits with code 2; a well-formed name that no run has matches nothing (with `--backends`, the script then lists the backends that have runs).

Anything else, or both flags together, prints the usage and exits with code 2. Each run's own
settings (samples, depths, shots) still come from its run folder; the argument only selects which folders.

**Collective plots** (`source/core/analysis/collective_plots.py`), written to `analysis/plots+summaries/<benchmark>_collective_<data types>/`:
`<benchmark>` is `mini`, `full`, the `<benchmark>` field of other benchmark arguments, or `any+nshots-N` for
`nshots_per_circuit=N` alone; `<data types>` joins the included kinds with `+` — `real-data`, `sim-qpu-noise`,
`sim-noiseless`, `sim-coherent` (`sim-qpu-noise` also stands for the `sim-fake-fez` runs that
`--include-simulations-real` adds) — e.g. `mini_collective_real-data+sim-noiseless/`.

| File | Shows |
|---|---|
| `awae_best_ems_vs_Nt.png` | AWAE vs $N_t$ of each run's best EMS at $N_{max}$ (solid) and at $N_{acc}$ (dashed) — one line if they coincide — with the threshold ε = 0.05 (`derived_quantities.EPSILON`) |
| `cumul_awae_best_ems_vs_Nt.png` | the same for the cumulative AWAE (Z_exact-weighted mean over the depths up to $N_t$, `derived_quantities.cumulative_awae()`), the quantity $N_{acc}$ is defined by |
| `awae_and_cumul_awae_best_ems_vs_Nt.png` | both side by side |
| `qpu_time.png` | `QPU_time_s` per run, log scale; bar labels and ticks in seconds (up to 60 s), minutes (up to 60 min), hours (up to 24 h) or days |
| `submission_time.png` | summed submission time per run (`timing_breakdown.cpu.submit_s` + `health_check_submit_s`, a part of `CPU_time_s`), log scale, units as above; runs without it (synchronous services, old runs) are left out |
| `collection_time.png` | `Collect_time_s` per run (summed result-collection time, failed jobs included), log scale, units as above; runs collected before it was timed are left out |
| `submission_to_results_duration.png` | `QPU_submission_to_results_duration_s` per run (earliest job submission to latest job result, wall clock, chunk, failed-attempt and health-check jobs), log scale, units as above; simulations are left out |
| `submission_to_results_duration_without_dead_time.png` | `QPU_submission_to_results_duration_without_dead_time_s` per run (the window minus long idle gaps), log scale, units as above; simulations and runs summarized without it are left out |
| `effective_cost_EUR.png` | `effective_cost_EUR` per run, log scale with € labels |
| `n_acc.png` | $N_{acc}$ per run, linear scale (a run with $N_{acc}$ = 0 shows a bar at 0); runs whose summary lacks it are left out |
| `e_min_nmax.png` | `E_min_Nmax` (the lowest AWAE over the schemes at $N_{max}$) per run, linear scale 0 to 1.05, with the threshold ε = 0.05 as a dashed line |

The same runs in one table, `<benchmark>_collective_<data types>.csv` in the same folder as the plots (e.g. `analysis/plots+summaries/mini_collective_real-data+sim-noiseless/mini_collective_real-data+sim-noiseless.csv`), which also holds a copy of each run's `summary_<run name>.json`: `;`-separated and padded with spaces so it also reads as a text table, UTF-8, one row per run, in the order of the plots; columns `run` (the plot label), `n_samples`, `sample_size`, `N_acc`, `best_scheme_Nacc`, `N_max`, `E_min_Nmax` (4 significant digits), `best_scheme_Nmax`, `QPU_time_s` (whole seconds), `QPU_time` (readable), `effective_cost_EUR` (2 decimals); a quantity the summary lacks is an empty cell.

Each run is labelled `<device_label>, layout <mode>, DD/MM/YYYY #N` (`device_label`, a key of the saved summary; older summaries on disk carry `computer_name`; the per-run figures use `figure_label_full`, §5.4:
e.g. `ibm-fez — real QPU, lnn-True`, or for a simulation `fake-fez — simulation (FakeFez noise), lnn-True`; a legacy summary without either uses the backend name).
An incomplete run also carries ` (n/N chunks)`. The shots (`<N> shots`) and finally the run folder are appended only where two runs would otherwise
share a label and that field tells them apart (a field the duplicates share is not appended). The plots read each run's
`summary*.json` and `run_data[_spam]*.json`, and also accept legacy summaries
(`effective_cost_euros`, best EMS without a `+SPAM` suffix — then the no-SPAM data is used); a
value a summary lacks is left out of that plot with a warning.

### 10.1 Run them in this order

These scripts are **not** all independent — `plot_awae.py` loads the summary
script's output, so it must run after it (`run_post_process.py` keeps this order):

```
qbcs.analyze_results()                        →  run_data.json (+ run_data_spam.json)
        │
        ├──────────────────────────────────────────────────────────┐
        ▼                                                            ▼
source/analysis_scripts/bacq_ps_qbcs_summary_json.py           source/analysis_scripts/plot_real_qpu_results.py
   →  summary.json                →  observable + max-error PNGs
        │
        ▼
source/analysis_scripts/plot_awae.py
   →  AWAE PNGs
```

`plot_real_qpu_results.py` only needs `run_data*.json` and can run any time
after `analyze_results()`. `plot_awae.py` opens
the run's summary (`processed_results/summary.json`, or `summary+nsamples-N.json` for a subset) unconditionally to read
`best_ems_Nmax` and `N_acc` ([plot_awae.py](source/analysis_scripts/plot_awae.py))
— it plots the accuracy threshold `N_acc` as a vertical line on every figure
and uses `best_ems_Nmax` to pick the isolated EMS curve in its third figure. Run
`bacq_ps_qbcs_summary_json.py` first, or `plot_awae.py` fails with
`FileNotFoundError` before drawing anything.

`source/analysis_scripts/plot_health_checks.py` is independent of the others (it needs only the health-check and chunk
records). Top panel: `P(011)` of each collected health check with the pass threshold marked; bottom panel:
UTC time bars of the health checks (orange, `H<n>`) and of the QPU jobs (blue, `J<n>` labelled by
the chunk numbers they contain), both labelled on the bottom axis, on a common axis ordered by submission time — a job's bar
spans the earliest chunk start to the latest chunk finish, so chunks that ran out of order inside a job are
visible. Finish times exist only after `qbcs_real_collect_*.py` has run (an uncollected health check is a red X and
an uncollected chunk job a purple X, both at their submission time; a hollow marker is for any other item
with a start time but no end time). `ZOOM_X` in the script restricts the axis for runs with hundreds of jobs (AQT).

### 10.2 The scripts

Every run records what it was — shots, samples, depths, `dt`, topology, SPAM — in `datafiles/<run>/run_info.json`,
written right after `build()` (`record_run_info()`, which calls `write_run_info()` in `source/core/records/run_info.py`). The scripts below read those settings from the run folder, so
the `RUN_FOLDER` in their headers only names the run when one is run standalone. Every run folder has one. `run_post_process.py` needs no per-QPU configuration: a `RUN_THESE` entry is a run name without its
`<benchmark>_` (e.g. `real-data_ibm-fez`, `sim-noiseless_local-aer_lnn-True`; the real QPUs are listed once in
`REAL_QPUS`, the simulations with a real QPU's calibration in `SIMULATED_QPUS` (`sim-qpu-noise_<provider>-<backend>`, so every layout is included), the local simulations in
`LOCAL_SIMULATIONS`, and all three lists go into `RUN_THESE`), cut off after any field, and
matches the one folder `<benchmark>_<entry>…` on disk (`<benchmark>` = `mini` or `full`), whose settings then come
from the run as above. If an entry matches several runs (e.g. two layouts or two dates), all are analyzed, in the order
QPU, layout, date, run number; to pick one, add fields or name it with
`dict(run=…, layout="optimized")` (or `"random"`, `"lowest"`, `"legacy"`, `"15-14-21"`) and/or `date="2026-10-11"` (or `"2026-10-11_#1"`). A run of another benchmark is
selected with `dict(run=…, benchmark=…)` (e.g. `benchmark="full+nshots-100"`), and an archived one with
`dict(run_folder=…)` (e.g. `run_folder="old_runs_full/full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-lowest_2026-09-21_#1"`).
Runs without a layout argument carry `layout-optimized` (scored; `layout-random` for AQT PIAST-Q and IQM EQE1 runs without `layout=` and `calibration_file=`), `layout-lowest`
(the lexicographically lowest SWAP-free placement) or `layout-legacy` (runs made before layouts were recorded; qiskit chose the qubits per circuit).

**A still-collecting run.** A plain entry uses every chunk collected so far: a complete, already analyzed run skips
the merge and analysis, and an incomplete one is analyzed as a `+nsamples-<collected>` subset (any uncollected chunks
and health checks are listed in it and warned about on the figures). `dict(run=…, n_samples=N)` analyzes exactly the
first N chunks (at most as many as are collected), e.g. `dict(run="real-data_ibm-fez", n_samples=120)`. A subset's outputs
are suffixed (`run_data+nsamples-120.json`, `summary+nsamples-120.json`), so the full run's own files are never touched.
Such a subset analysis varies only the RC samples: its SPAM correction uses the calibration of every collected chunk
(`spam_confusion_matrix_inv.npy`); `analyze_results(n_samples=N, spam_n_samples=M)` with M below the collected chunks
uses the first M instead (`spam_confusion_matrix_inv+nsamples-M.npy`). `run_data_spam*.json` records the chunks used as `spam_calibration_samples`.

`run_info.json` is written right after `build()`, before the pre-submission health check
(`qbcs.record_run_info(run_folder)`, also called by `resolve_physical_qubits(run_folder=…)`), so a run
that the check aborted is still on record. For such a run — health checks but no submitted chunks —
`run_post_process.py` warns that the health check found the QPU degraded and the benchmark was not run,
and produces only the health-check plot and `processed_results/health_checks.json` (`health_checks+nsamples-N.json` for a subset analysis).

| Script | Reads | Writes | Config you must set |
|---|---|---|---|
| `source/analysis_scripts/bacq_ps_qbcs_summary_json.py` | `processed_results/run_data_spam.json` if the run has `do_SPAM` (plus `run_data.json`), else `run_data.json`; `job_records/job_record_chunk_*.json`; `health_checks/*.json`; this run's `timestamps_utc.csv`; the *shared* `datafiles/circuits/<topology>_Nt<steps>/timestamps_utc.csv` (or `timestamps_qiskit_v13_qpy_utc.csv` for all-to-all runs, §5.4) — the `_local_time.csv` logs for runs made before the UTC logs existed | `processed_results/summary.json` (see *Timing in the summary* and *Cost in the summary* below) | `RUN_FOLDER` (standalone); a per-backend metadata block (`QPU_PROVIDER`, `ARCHITECTURE`, pricing model, …) is hardcoded in an `if BACKEND_NAME == …` chain on the run's `<provider>-<backend>` (e.g. `ibm-fez`) — a new backend needs a new `elif` |
| `source/analysis_scripts/plot_awae.py` | `processed_results/run_data.json` (+ `run_data_spam.json` if `do_SPAM`); `processed_results/summary.json` | `analysis/plots+summaries/<run_folder>/awae_no_spam.png` (+ `awae_spam.png`, `awae_best_ems_spam_no_spam_comparison.png` if `do_SPAM`) | `RUN_FOLDER` (standalone) |
| `source/analysis_scripts/plot_awae_nsamples_comparison.py` | `processed_results/run_data[_spam][+nsamples-N].json` for each sample count in `NSAMPLES_LIST` (each analyzed beforehand, e.g. by `run_post_process.py --nsamples-comparison`); the reference analysis's `summary` (`best_ems_Nmax`, `N_acc`) | `analysis/plots+summaries/<run_folder>/awae_best_ems_nsamples_comparison.png` | `RUN_FOLDER`, `N_SAMPLES` (reference analysis; `None` = full run), `NSAMPLES_LIST` (standalone; the environment variable `QPU_NSAMPLES_LIST` overrides it) |
| `source/analysis_scripts/plot_real_qpu_results.py` | `processed_results/run_data.json` (+ `run_data_spam.json` for a SPAM run); the exact reference is the `perfect_circuit_obs_values` stored in `run_data` | `analysis/plots+summaries/<run_folder>/obs_lnn{LNN}_ns{NSHOTS}_{no_spam,spam}.png` (2×4 grid); `max_error_lnn{LNN}_ns{NSHOTS}.png` (2×3 max-error grid) | `RUN_FOLDER` (standalone) |
| `source/analysis_scripts/plot_health_checks.py` | `health_checks/health_check_*.json`; `job_records/job_record_chunk_*.json` (older local simulator runs: `chunk_times_UTC_time.json`) | `analysis/plots+summaries/<run_folder>/health_checks[+nsamples-N].png`; `processed_results/health_checks[+nsamples-N].json` | `RUN_FOLDER` (standalone; skips runs without new-format health checks) |

**Timing in the summary** (`timing_definitions_version: 3`; definitions in `source/core/records/job_timing.py`,
aggregation in `source/core/analysis/run_timing.py: aggregate_run_timing`). Health checks that accompany the analysed chunks
are included in all of them:

- `CPU_time_s`: circuit construction (myQLM) + conversion to Qiskit + transpilation (chunks, SPAM, health
  checks) + submission. The build is scaled to the analysed subset. The health check circuit is transpiled once
  per run, and its time (`transpiled_circuits/health_check_transpiled.json`) is counted whole, also in a subset
  analysis (runs without that file: the sum over the health check records in scope). Chunks whose transpilation and
  submission were not logged separately count their whole `qpu_chunk` interval. The analysis also reads the `qpu_chunk<nc>_start/_finished` pair (only for runs without transpile/submit
  labels) and the `chunk<a> to chunk<b>_start/_end` pair of a batched job; `Starting chunk submission.` marks
  the start of a submission session and `All chunks submitted.` its end. All other labels (AQT
  `qpu_chunk<nc>_part<p>_*`, health-check and build labels) are informational and not read by the analysis.
  Converted old AQT runs did not log transpilation: it is estimated as the measured transpilation time
  per circuit of the AQT runs that did log it (`run_timing.aqt_transpile_rate`) × the run's chunk
  circuits (+ the SPAM circuits once), and is 0 while no such run exists; the terminal says which runs
  the rate came from, and `timing_breakdown.cpu.transpile_estimate` records it.
- `QPU_time_s`: billed time (IBM, IQM Resonance); for IQM VLQ the Resonance equivalent, `ceil(compilation +
  execution)` per job from IQM's server timeline; 0.9 s per circuit (IQM EQE1); AQT per circuit by its
  shot count — 3 s up to 10 shots, 15 s at 200 (AQT's maximum), linear in between
  (`job_timing.aqt_qpu_s_per_circuit`), for chunks and health checks alike. A job above 200 shots gets
  15 s per circuit and a printed warning. Failed jobs that reached the
  provider are included (§7).
- `QPU_execution_time_s`: the QPU's execution time alone (no compilation, set-up or billing rounding), summed
  over the run's jobs, where the provider reports it: IQM VLQ `remote_hw_runtime` (what VLQ charges the
  project allocation), IQM Resonance `metrics.execution_seconds`, IBM
  `circuits_execution_time_ns`. It is `"N/A"` for IQM EQE1, AQT and simulations, and whenever
  any job of the run lacks it.
- `Queue_time_s`: per job, end of submission → start of execution (table in §7); for IQM VLQ, end of the upload
  → IQM's compilation start, which includes the wait in IT4I's HEAppE job system. IBM, IQM Resonance, IQM VLQ and
  IQM EQE1 queue many jobs at once, so the waits overlap: the run's queue time is the longest single wait over
  all chunk jobs and health checks, failed jobs included. Synchronous services (AQT, the local simulator) run their jobs one after
  another, so there the waits are added up.
  `timing_breakdown.queue.sum_of_jobs_s` keeps the plain sums.
- `QPU_submission_to_results_duration_s`: `date_qc_end − date_qc_start`, from the earliest job submission
  to the latest job result, over the chunk jobs, the failed attempts and the health checks.
- `QPU_submission_to_results_duration_without_dead_time_s`: that window minus the gaps between the merged job
  intervals that are longer than 30 minutes (2 hours on AQT), e.g. a failure noticed hours later and resubmitted.
- `Collect_time_s`: collecting the results — from a finished job to its saved counts: downloading the
  result (the first chunk of a job shared by several chunks carries the download), downloading the executed
  circuits for the circuit check, saving counts and records. Waiting for unfinished jobs is not included.
  Chunks: the `chunk{nc}_collect_start/_end` labels of the run's timestamp log; health checks: their
  `timing.collect_s` (label `health_check_<n>_collect_*`). AQT and the local simulator return results
  synchronously, so theirs is the handling of the received results. `"N/A"` for runs collected before it was
  timed (then 0 in the total); `timing_breakdown.collect` counts the chunks and checks without a measurement.
  Failed jobs' collection is included (§7).
- The same per EMS (`*_per_EMS_s`), `FT_clas + FT_quant + FT_queue + FT_collect = 1`, and
  `T_generation_submission_execution_s` = CPU + QPU + queue + collection. Summaries of version 1
  have no queue term and those below version 3 no collection term, so their `FT_*` and total are not comparable.
- `completeness`: the chunks analysed, and the chunks and health checks of the run not collected yet (see
  *Incomplete runs* below).
- `timing_breakdown` holds every part and the source of each value; `timing_notes` explains them;
  `circuit_verification` holds the circuit checks (§7).
- `QPU_time_s` is the successful jobs' QPU time plus the failed jobs' (§7); `Queue_time_s` and
  `Collect_time_s` include the failed jobs too.
  Everything derived from `QPU_time_s`, including the cost, uses this value.

**Incomplete runs.** The collect scripts do not wait: they collect the jobs that have finished, in
whatever order they finished, leave the others for a later run of the script, and mark failed jobs failed
(the same for health checks; a submission's pre-submission check is still waited for). A run can therefore
have uncollected chunks in the middle. It is analysed all the same (`source/analysis_scripts/run_post_process.py`, or
`analyze_results()` in the collect script) from the chunks collected so far: an analysis of *n* chunks
covers the first *n* collected ones (`source/core/records/chunk_scope.py`), everywhere — results, timing,
cost, circuit checks — except the SPAM correction, which uses every collected chunk's calibration. Its files carry the number of chunks in their name as `+nsamples-<n>` (`run_data+nsamples-<n>.json`,
`health_checks+nsamples-<n>.png/.json`, `summary+nsamples-<n>.json`, as for a subset analysis, §5.4); their `completeness` block lists the uncollected chunks and
health checks, and every figure carries a red warning line naming them. Merged results keep each chunk's
own number, which the NEC analysis needs (a sample's NEC input state is its chunk number mod 8). In the
health check plot, an uncollected health check is a red cross and an uncollected chunk job a purple cross,
both at their submission time.

**Cost in the summary.** The benchmark's runs used free access, so `actual_cost_EUR` is 0. `effective_cost_EUR` is what the
run would have cost at public list prices. It is computed from the run's own records (`source/core/analysis/run_timing.py: qpu_workload`):

- jobs: chunks sharing a job id count once;
- circuits: each chunk's circuits plus its SPAM circuits (68 in the full benchmark);
- shots;
- health checks, which are included and count as one job each;
- failed jobs, and their circuits and shots (results lost, not counted as jobs above): added to the cost of every backend and named in `cost_explanation`, by the rules in §7 (AQT: the circuits finished before the failure; the other backends: failed jobs that reached the provider, in full);
- submissions that may have run on the provider and jobs never collected: not counted, named in `cost_explanation` with their cost unknown.

`cost_explanation` gives the reason the actual cost is 0, the pricing model with links, and the result.
VAT is excluded. The prices and the exchange rate are hard-coded at the top of the script, with the date they
were looked up (2026-09-27; the IQM Resonance credits, used for the VLQ/Sirius estimate, 2026-09-28).

| Backend | Priced as | Fields |
|---|---|---|
| IBM | `QPU_time_s` (failed jobs included) × $96 per minute (IBM Quantum Pay-As-You-Go) | `effective_cost_USD`, `exchange_rate_EUR_per_USD`, `effective_cost_EUR` |
| IQM Resonance Emerald, IQM EQE1 (the same model) | Amazon Braket, IQM Emerald: $0.30 per task + $0.0016 per shot. One IQM job (≤ 100 circuits on Resonance, ≤ 70 on EQE1) is one AWS program set, billed as one task. | as for IBM |
| IQM Resonance Garnet | the same, with $0.00145 per shot | as for IBM |
| IQM VLQ | a synthetic estimate combining the IQM Resonance and Amazon Braket billing schemes: $0.30 per task + $0.00064 per shot (Braket's Emerald price × Resonance's credits per second for Sirius, a Star 24 machine like VLQ, 0.3, over Emerald's, 0.75). A job holds up to 12 chunks (`VLQ_CHUNKS_PER_JOB`), so it counts as the 100-circuit program sets its whole chunks fill (5 chunks of 20 circuits in the mini benchmark, 1 of 68 in the full one); a health check is one task; failed jobs count the same way. | as for IBM |
| IQM Resonance Sirius | Braket does not offer Sirius, a Star 24 machine like VLQ: the same synthetic estimate as IQM VLQ | as for IBM |
| AQT PIAST-Q | Scaleway, AQT IBEX-Q1: €0.30 per circuit + €0.021 per shot | `effective_cost_EUR` only |
| simulations (`sim-*` data types) | no QPU | `effective_cost_EUR` = 0 |

### 10.3 Things that will trip you up

- **Figures land in `analysis/plots+summaries/<run_folder>/`** (auto-created; the collective plots in
  `analysis/plots+summaries/<benchmark>_collective_<data types>/`), *not* inside the run's
  own `datafiles/<run_folder>/`. `summary.json` is the one
  exception — it's written back into that run's `processed_results/` (and copied as `summary_<run name>.json` into the figures' folder).
- **Run standalone, a script analyzes the run named by its `RUN_FOLDER` constant.**
  Get it wrong and the script reads another run's folder (if it happens to exist)
  or stops with `FileNotFoundError`. Once the folder is found, every setting comes
  from its `run_info.json`. `run_post_process.py` avoids this by passing the folder
  explicitly (`QPU_RUN_FOLDER`).

---


