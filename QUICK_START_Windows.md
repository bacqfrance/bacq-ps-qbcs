# Quick start (Windows) — from a fresh clone to benchmark runs on every QPU

The shortest path through the project, step by step. Details are in [`USER_MANUAL.md`](USER_MANUAL.md) and in each
QPU's own readme (`source/benchmark_scripts/real_qpu_scripts/<QPU>/readme_*.md`).

All commands are for **Windows Terminal** with its default **PowerShell** profile, run from the project
root (`cd "<path to>\Quantum_Benchmarking_project"`). Every script also moves to the project root by
itself, so it can be started from any other directory. On Linux/macOS, use
[`QUICK_START_Linux.md`](QUICK_START_Linux.md) instead. Windows Terminal ships with Windows 11; on
Windows 10, install it from the [Microsoft Store](https://aka.ms/terminal).

> ⚠️ **Cost.** Real-QPU runs (steps 3, 5 and 6) can cost a significant amount of money — up to
> ~€18,000 for the full benchmark on AQT. The expected costs below are at **public list prices, looked up on
> 2026-09-27 (the IQM Resonance credits, used for the VLQ/Sirius estimate, on 2026-09-28), VAT excluded**, with $1 = €0.8778:
>
> | QPU | List price used |
> |---|---|
> | IBM | $96 per minute of billed QPU time (IBM Quantum Pay-As-You-Go) |
> | IQM Resonance Emerald, IQM EQE1 | $0.30 per task + $0.0016 per shot (Amazon Braket, IQM Emerald); one job (≤ 100 circuits on Resonance, ≤ 70 on EQE1) = one task |
> | IQM Resonance Garnet | $0.30 per task + $0.00145 per shot (Amazon Braket, IQM Garnet) |
> | IQM VLQ (IT4I), IQM Resonance Sirius | synthetic: $0.30 per task + $0.00064 per shot (Braket's Emerald price × IQM Resonance's Sirius/Emerald credit ratio, 0.3/0.75); each 100-circuit program set of whole chunks = one task |
> | AQT PIAST-Q | €0.30 per circuit + €0.021 per shot (Scaleway, AQT IBEX-Q1) |
>
> Your actual cost depends on your contract (free allocations, research access, changed prices), so
> recompute it before each real-QPU step from the base numbers given for that step: **jobs, circuits,
> shots, and QPU time**. After a run, `source/analysis_scripts/run_post_process.py` writes the list-price estimate
> (`effective_cost_EUR`) into the run's `summary.json` (USER_MANUAL §10.2).
>
> 🆓 **IBM Quantum Open Plan.** IBM's free Open Plan gives 10 minutes of QPU time per month. That is enough
> to run all the IBM examples in this guide — the health check (seconds), the mini benchmark (well under
> a minute) and the full benchmark (~2 min) — a few times each month at no cost. Use it for the IBM steps
> if you have no paid IBM plan. An IBM *instance* is a plan-bound resource of your IBM Quantum Platform account (its name is shown in the
> platform's Instances list; an Open Plan account has the free `open-instance`). The IBM scripts use your account's
> Open Plan instance by default and never pick a paid instance on their own; `instance=NAME` selects another one
> (then the run is billed at the list price).
>
> 🆓 **IQM Resonance free allocation.** IQM Resonance gives 30 credits per month for free — enough for the
> mini benchmark on Emerald, Garnet and Sirius (§5.2), not for the full one.
>
> 🆓 **Free access through EuroHPC.** IQM EQE1 (LRZ), IQM VLQ (IT4I) and AQT PIAST-Q (PCSS) are EuroHPC
> machines: running on them is free with access granted through EuroHPC. See
> <https://www.eurohpc-ju.europa.eu/eurohpc-quantum-computers/our-quantum-computers_en>.
>
> **Times.** The circuit-generation times for the full benchmark below were measured on a desktop-class Windows PC. QPU times are measured on past
> runs (IBM, IQM Resonance: billed time),
> modelled (IQM EQE1: 0.9 s per circuit) or estimated (IQM VLQ: execution ≈ 3 ms per circuit + 0.3 ms per shot; AQT: 3 s per circuit at
> ≤ 10 shots, 15 s at 200 shots). Queue time comes on top of the QPU time and is unpredictable: from minutes
> to days on a busy machine.
>
> The mini-benchmark times in §5.2 come from runs of 2026-10-02 (AQT's QPU time is estimated; its 5 h wallclock includes the queue).

## Contents

1. [Install Python 3.12](#1-install-python-312)
2. [Install the virtual environments](#2-install-the-virtual-environments)
3. [Add the QPU tokens and run a health check on each QPU](#3-add-the-qpu-tokens-and-run-a-health-check-on-each-qpu) — 💰 real hardware
4. [See the workflow with the mini benchmark on the local simulator](#4-see-the-workflow-with-the-mini-benchmark-on-the-local-simulator-no-cost) — local, no cost
5. [Run the mini benchmark on every QPU](#5-run-the-mini-benchmark-on-every-qpu)
   1. [Build the mini circuits](#51-build-the-mini-circuits-local-local_simvenv-no-cost) — local, no cost
   2. [Submit and collect](#52-submit-and-collect) — 💰 real hardware
   3. [Analyze](#53-analyze-local_simvenv)
6. [Run the full benchmark on every QPU](#6-run-the-full-benchmark-on-every-qpu)
   1. [Build the full circuits](#61-build-the-full-circuits-local-local_simvenv-no-cost--but-hours-of-cpu-time) — local, no cost, hours of CPU time
   2. [Optional rehearsal without QPU time](#62-optional-rehearsal-without-qpu-time-local-no-cost) — local, no cost
   3. [Submit and collect](#63-submit-and-collect) — 💰 real hardware
   4. [Analyze](#64-analyze-local_simvenv)

---

## 1. Install Python 3.12

The project needs **Python 3.12**: the pinned package versions are built for it.

1. Download the latest 3.12 release's **Windows installer (64-bit)** from
   <https://www.python.org/downloads/windows/>.
2. Run it and tick **"Add python.exe to PATH"** on the first page, then click *Install Now*. This also
   installs the `py` launcher, which the commands in this guide use.
3. Open a new Windows Terminal tab and check:

   ```powershell
   py --version        # should print Python 3.12.x
   ```

You can try another Python version, but the pinned package versions may not install or may behave
differently on it — if you hit errors, switch to 3.12 first. Several versions can be installed side by
side; if `py --version` shows another one, write `py -3.12` instead of `py` in the `venv` commands of §2.

---

## 2. Install the virtual environments

PowerShell blocks the venv activation scripts by default. Allow local scripts once, for your user only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

One venv per backend, all under `venvs/` (git-ignored). Build each one in a single pass
from its pinned requirements file — don't layer one venv on top of another.

```powershell
# base pipeline: circuit building (myQLM), local simulation, analysis
py -m venv venvs\local_sim.venv
.\venvs\local_sim.venv\Scripts\Activate.ps1
pip install -r venvs\requirements_local_sim.txt
deactivate

# one per QPU (no myQLM in any of these)
py -m venv venvs\ibm.venv
.\venvs\ibm.venv\Scripts\Activate.ps1
pip install -r source\benchmark_scripts\real_qpu_scripts\IBM\requirements_ibm.txt
deactivate

py -m venv venvs\iqm_eqe1.venv
.\venvs\iqm_eqe1.venv\Scripts\Activate.ps1
pip install -r source\benchmark_scripts\real_qpu_scripts\IQM_EQE1\requirements_iqm_eqe1.txt
deactivate

py -m venv venvs\iqm_resonance.venv
.\venvs\iqm_resonance.venv\Scripts\Activate.ps1
pip install -r source\benchmark_scripts\real_qpu_scripts\IQM_RESONANCE\requirements_iqm_resonance.txt
deactivate

py -m venv venvs\iqm_vlq.venv           # qiskit 2.1 — pinned by IT4I's qaas client
.\venvs\iqm_vlq.venv\Scripts\Activate.ps1
pip install -r source\benchmark_scripts\real_qpu_scripts\IQM_VLQ\requirements_iqm_vlq.txt --extra-index-url https://opencode.it4i.eu/api/v4/projects/107/packages/pypi/simple
deactivate

py -m venv venvs\aqt.venv            # runs qiskit 1.4.6, not 2.5.x — required by qiskit-aqt-provider
.\venvs\aqt.venv\Scripts\Activate.ps1
pip install -r source\benchmark_scripts\real_qpu_scripts\AQT_PIAST-Q\requirements_aqt.txt
deactivate
```

Leave a venv with `deactivate`. Keep myQLM out of the QPU venvs: circuits are built in `local_sim.venv` and read from the cache.

**Check the base install** (recommended; `local_sim.venv`, 5 to 10 minutes on a recent PC):

```powershell
.\venvs\local_sim.venv\Scripts\Activate.ps1
pytest source\tests\
```

All tests should pass. The tests build their circuits in `datafiles/tests/circuits/` and never touch your
own circuit caches (USER_MANUAL §9.1). `test_12` also needs `aqt.venv` (step 2 above).

---

## 3. Add the QPU tokens and run a health check on each QPU

Save each token as a plain-text file in the `tokens/` folder (git-ignored; the scripts read them with
`source/core/qpu/tokens.py: read_token()`, see `tokens/README.md`). In PowerShell, for example:

```powershell
Set-Content -Path tokens\ibm_token -Value "<your IBM token>" -NoNewline -Encoding ascii
```

Repeat with the file name from the table for each QPU you have access to (`tokens/README.md` says what
each file holds and where to get it):

| QPU | Token file (in `tokens/`) | Health-check script (in `source/benchmark_scripts/real_qpu_scripts/`) | venv |
|---|---|---|---|
| IBM | `ibm_token` | `IBM/health_check_real_ibm.py` | `ibm.venv` |
| IQM EQE1 (LRZ, via MQSS) | `iqm_eqe_token` | `IQM_EQE1/health_check_real_iqm_eqe1.py` | `iqm_eqe1.venv` |
| IQM Resonance (Emerald) | `iqm_resonance_token` | `IQM_RESONANCE/health_check_real_iqm_resonance.py` | `iqm_resonance.venv` |
| IQM VLQ (IT4I, via LEXIS) | `iqm_vlq_token` — written by the first run: it opens the LEXIS (MyAccessID) login in the browser once, then reuses and rotates the saved token; also create `iqm_vlq_project_info` with the lines `PROJECT = …` and `RESOURCE = …` (your LEXIS project and its VLQ resource, see `tokens/README.md`) | `IQM_VLQ/health_check_real_iqm_vlq.py` | `iqm_vlq.venv` |
| AQT PIAST-Q (PCSS) | `aqt_piast-q_token` | `AQT_PIAST-Q/health_check_real_aqt.py` | `aqt.venv` |

Every script needs the QPU named on the command line with `backend=NAME` — there is no default. The
allowed names are in the `backends.json` next to the scripts (IBM: `ibm_marrakesh`, `ibm_fez` and `ibm_kingston` on the free Open Plan instance, which the IBM scripts use unless
given `instance=NAME`, and `ibm_boston`, `ibm_pittsburgh`, `ibm_aachen` and `ibm_phoenix`, which are listed only on paid
instances; IQM Resonance: `emerald`, `garnet`, `sirius`; IQM EQE1: `EQE1`; IQM VLQ: `VLQ`; AQT: `PIAST-Q`). Run it in its venv, e.g.

```powershell
.\venvs\ibm.venv\Scripts\Activate.ps1
python source\benchmark_scripts\real_qpu_scripts\IBM\health_check_real_ibm.py backend=ibm_marrakesh
```

Each script submits a 3-qubit reference circuit, waits for the result and prints `P(011)`: **> 0.5 means
the QPU is healthy**. This also confirms that the token, the venv and the connection work.

> 💰 **COST — each health check runs on real hardware.** Per QPU and per check: 1 job, 1 circuit, 100 shots;
> QPU time a few seconds (AQT: ~9 s estimated; EQE1: 0.9 s); the wait is dominated by the queue.
> Expected list-price cost per check: IBM a few euros (a few billed seconds at $96/min; free on the Open
> Plan), IQM EQE1 and IQM Resonance ≈ €0.40 each, AQT ≈ €2.40.

---

## 4. See the workflow with the mini benchmark on the local simulator (no cost)

The mini benchmark (8 samples, Nt = 1, 3, 200 shots per circuit) is what every benchmark script in
`source/benchmark_scripts/` runs with the `mini` argument. On the local Aer simulator it shows the whole
pipeline — build the circuits, health check, submit, analyze — in about a minute. Run these in this order,
from the project root (the scripts are in `source/benchmark_scripts/local_simulation_scripts/`):

| # | Script | venv | CPU time |
|---|---|---|---|
| 1 | `benchmark_noiseless_all_to_all.py` — the whole pipeline, noiseless; AWAE is shot noise only. Builds the circuits for AQT, IQM VLQ and IQM Resonance Sirius | `local_sim.venv` | ~10 s including the circuit build |
| 2 | `benchmark_noiseless_lnn.py` — the same with linear-nearest-neighbor circuits. Builds the circuits for IBM, IQM EQE1 and IQM Resonance Emerald / Garnet | `local_sim.venv` | ~10 s including the circuit build |
| 3 | `benchmark_fake_fez_lnn.py` — #2 under IBM Fez noise, transpiled onto Fez's qubits; needs #2 first | `ibm.venv` | ~35 s with `scheduling_simulation=false` |

```powershell
.\venvs\local_sim.venv\Scripts\Activate.ps1
python source\benchmark_scripts\local_simulation_scripts\benchmark_noiseless_all_to_all.py mini
python source\benchmark_scripts\local_simulation_scripts\benchmark_noiseless_lnn.py mini
deactivate

.\venvs\ibm.venv\Scripts\Activate.ps1        # #3: FakeFez comes with qiskit-ibm-runtime
python source\benchmark_scripts\local_simulation_scripts\benchmark_fake_fez_lnn.py mini
deactivate
```

Script #3 takes the optional `scheduling_simulation=true|false`, default `true` (the other two take no such argument): `true` transpiles the circuits scheduled, so idle qubits relax through T1/T2 as on the QPU (full benchmark ≈ 4 h once the circuits exist); `false` skips the scheduling, so idle relaxation is **not** taken into account (only the relaxation during gates), like the noiseless and coherent-noise simulations, and runs in ≈ 1 h. The choice is recorded in `run_info.json`, and a run folder cannot be resumed with the other value (USER_MANUAL §4.2).

Each script prints its build, execution, analysis and total wall times. Results land in
`datafiles/mini_sim-noiseless_local-aer_lnn-False_service-local_spam-True_layout-none_<YYYY-MM-DD>_#1/` (the UTC date of the run and its number that day), the same with
`lnn-True`, and `mini_sim-fake-fez_local-aer_lnn-True_…` (USER_MANUAL §5.4 explains the names);
the first circuit of every error-mitigation role (before transpilation; also after it in the noisy fake-Fez run), is saved as PDFs in
their `circuit_drawings/` subfolder. A rerun continues the latest run folder of the same settings while it is unfinished;
once it is finished, the next run starts a new folder dated today (UTC) with the next run number of the day.

---

## 5. Run the mini benchmark on every QPU

The mini benchmark is a cheap end-to-end rehearsal of a hardware run: **8 chunks (RC samples), 2 depths
(Nt = 1, 3), 200 shots per circuit**. One chunk = 12 benchmark circuits + 8 SPAM circuits = 20 circuits.

### 5.1 Build the mini circuits (local, `local_sim.venv`, no cost)

The QPU venvs have no myQLM, so the circuits must be built first: scripts #1 and #2 of §4 do it
(#2 for IBM, IQM EQE1 and IQM Resonance Emerald / Garnet, #1 for AQT, IQM VLQ and IQM Resonance Sirius). Run them first if you haven't.
Each also runs a noiseless simulation, whose AWAE should be ≈ 0 (shot noise only). The mini circuit caches take ≈ 5 MB each, and every mini run folder ≈ 2–3 MB.

### 5.2 Submit and collect

The scripts are in `source/benchmark_scripts/real_qpu_scripts/<QPU>/`. Run them with `mini` for the mini
benchmark; without arguments they only print their usage (USER_MANUAL §4.3 lists the arguments).

> 💰 **COST — this step runs on real hardware.** Expected cost at list prices, health checks included
> (base numbers — jobs, circuits, shots, QPU time — in the table below it):
>
> | QPU | Calculation | Expected cost |
> |---|---|---|
> | IBM | 15 s billed (measured) × $96/min = $24 | ≈ €21 (free on the Open Plan) |
> | IQM EQE1 | 5 tasks × $0.30 + 32,200 shots × $0.0016 = $53.02 | ≈ €47 |
> | IQM Resonance (Emerald, Garnet) | 4 tasks × $0.30 + 32,200 shots × $0.0016 = $52.72 (Emerald; Garnet: $0.00145 per shot, $47.89) | ≈ €46 (Emerald), €42 (Garnet) |
> | IQM Resonance Sirius | 4 tasks × $0.30 + 32,200 shots × $0.00064 (synthetic price, as for VLQ) = $21.81 | ≈ €19 |
> | IQM VLQ | 4 tasks (2 for the job's 8 chunks at 5 chunks per task, 2 health checks) × $0.30 + 32,200 shots × $0.00064 = $21.81 | ≈ €19 |
> | AQT PIAST-Q | 169 executions × €0.30 + 32,900 shots × €0.021 | ≈ €740 |
>
> 🆓 **IBM Quantum Open Plan.** The mini benchmark takes 15 s of the Open Plan's 10 free minutes per month
> on any IBM machine (Fez, Kingston, Marrakesh), so it costs nothing there.
>
> 🆓 **IQM Resonance free allocation.** IQM Resonance gives 30 credits per month for free. The mini
> benchmark fits within it on every machine used here: **Emerald 13.5 credits, Garnet 7, Sirius 4.5**
> (so all three together take 25 of the 30).
>
> 🆓 **EuroHPC: EQE1, VLQ and PIAST-Q are free** with access through EuroHPC
> (<https://www.eurohpc-ju.europa.eu/eurohpc-quantum-computers/our-quantum-computers_en>), so the costs of
> their rows below are list-price estimates only.

Every submit/collect/simulate/status script needs the QPU named on the command line with
`backend=NAME` — e.g. `backend=ibm_marrakesh` (IBM), `backend=emerald` (IQM Resonance), `backend=EQE1`,
`backend=VLQ`, `backend=PIAST-Q` — and stops before connecting if it's missing or not listed in the
`backends.json` next to the scripts. That file also sets, per backend, whether it needs linear-nearest-neighbour
or all-to-all circuits (`"lnn"`), so `LNN` follows the backend automatically; to use another device, add
it there with its real topology. Submit/collect/status must use the same `backend=` for a given run —
the run folder name on disk is derived from it.
Collect with the same arguments as the submission (here: `backend=` and `mini`); with `layout=` or `date=` given, collect reads the latest run of
that layout, or the one of `date=YYYY-MM-DD` or `date=YYYY-MM-DD_#N` (the run folder's name ends in the UTC day of its submission and the run number of that day). Rerunning
a submit script continues the latest run of its settings while it is unfinished, else starts a new dated one (`--force-new-run` starts a new one regardless). Each submit
script runs a PRE health check, aborts if the QPU is degraded, submits, and submits a POST health check.

| QPU | venv | Commands | Jobs | Circuits (executions) | Shots | QPU time |
|---|---|---|---|---|---|---|
| IBM | `ibm.venv` | `python ...\IBM\qbcs_real_submit_ibm.py backend=ibm_marrakesh mini`,<br>later `python ...\IBM\qbcs_real_collect_ibm.py backend=ibm_marrakesh mini` | 1 (all 8 chunks) + 2 health checks | 160 + 2 | 32,000 + 200 | **15 s billed**, measured on Fez, Kingston and Marrakesh (QPU execution alone ~8.5 s) |
| IQM EQE1 | `iqm_eqe1.venv` | `python ...\IQM_EQE1\qbcs_real_submit_iqm_eqe1_with_retries.py backend=EQE1 mini`,<br>later `python ...\IQM_EQE1\qbcs_real_collect_iqm_eqe1.py backend=EQE1 mini` | 3 (≤ 70 circuits each: 3 + 3 + 2 chunks) + 2 | 160 + 2 | 32,000 + 200 | **~146 s** modelled (0.9 s per circuit × 162 circuits; EQE1 reports no QPU time) |
| IQM Resonance | `iqm_resonance.venv` | `python ...\IQM_RESONANCE\qbcs_real_submit_iqm_resonance.py backend=emerald mini`,<br>later `python ...\IQM_RESONANCE\qbcs_real_collect_iqm_resonance.py backend=emerald mini` | 2 (≤ 100 circuits each: 5 + 3 chunks) + 2 | 160 + 2 | 32,000 + 200 | **billed: Emerald 18 s, Garnet 14 s, Sirius 15 s** (QPU execution alone: 14, 10 and 11 s) |
| IQM VLQ | `iqm_vlq.venv` | `python ...\IQM_VLQ\qbcs_real_submit_iqm_vlq_with_retries.py backend=VLQ mini`,<br>later `python ...\IQM_VLQ\qbcs_real_collect_iqm_vlq.py backend=VLQ mini` | 1 (all 8 chunks) + 2 | 160 + 2 | 32,000 + 200 | **15 s** (IQM Resonance-equivalent billed time; QPU execution alone ~12 s) |
| AQT PIAST-Q | `aqt.venv` | on the PCSS cluster (Linux), `sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop.sbatch backend=PIAST-Q mini` \* (synchronous, no collect step) | 8 chunks (1 batch each) + 9 health checks (1 PRE + 1 after each chunk) | 160 circuits, **160 executions** (200 shots fit AQT's limit of 200 per circuit, so each chunk runs once) + 9 | 32,000 + 900 | **~41 min** estimated (160 executions × 15 s at 200 shots = 40 min, plus 9 health checks ≈ 1¼ min; AQT reports no QPU time), measured on 2026-10-02: 5 h wallclock including queue |

`...` = `source\benchmark_scripts\real_qpu_scripts`.

\* The PIAST-Q run was done through Slurm on the PCSS Quantum UI cluster (see below). Without Slurm, run the loop script directly: `python ...\AQT_PIAST-Q\qbcs_real_submit_aqt_loop.py backend=PIAST-Q mini`.
It then stays up for the whole run and takes **1.5–2.5 times longer** per circuit than on the PCSS cluster (probably less efficient network communication), so Slurm is the recommended way.

**Qubit layout.** Every submit and simulate command also takes `layout=` — `optimized` (the default for IBM, IQM Resonance and
IQM VLQ), `random`, `lowest`, or the qubits themselves, e.g. `layout=15,14,21` (USER_MANUAL §5.4) — and the run folder name ends
in it, followed by the date (`mini_…_layout-optimized_2026-10-11_#1`). **IQM EQE1 and AQT PIAST-Q report no calibration data, so without a calibration file they run on a
random layout** (a notice is printed). To optimize their layout instead, pass a provider calibration file from the `calibration_input_files/` folder, for example:

```
layout=optimized calibration_file=calibration_input_files/Euro-Q-Exa/calibration_data_Euro-Q-Exa_20261020_active_qubits.json    # EQE1
layout=optimized calibration_file=calibration_input_files/PIAST-Q/piastq_characterisation_2026-09-29.json                        # PIAST-Q
```

`layout=optimized` for these two without `calibration_file=` is an error. Their **simulate** scripts (§6.2) require `layout=` to be
named — they take the same calibration file as noise data and have to be told which real-submission behaviour to mimic
(`layout=random` is what a submission without a file does). Collect commands take `layout=` too; it is optional: without `layout=` and `date=` every run of that benchmark with something left to collect is collected (or reported) in turn, all layouts and dates; give `layout=` to pick one run (including an already finished one); `date=YYYY-MM-DD` alone takes each layout's latest run of that day, `date=YYYY-MM-DD_#N` each layout's run with that number (layouts without one are skipped).

**Wall time of the whole mini run** (circuit loading and transpiling, submission, queue, execution and collection; measured 2026-10-02, health checks included):

| QPU | Wall time | Of which |
|---|---|---|
| IBM | 2–2¾ min (Kingston 117 s, Marrakesh 144 s, Fez 159 s), mostly depending on the queue; a failed job that must be resubmitted adds more | CPU ~31 s, queue 24–65 s, collecting ~47 s |
| IQM EQE1 | ~3¾ min (221 s) | CPU 29 s, queue 35 s |
| IQM Resonance | 65–79 s (Sirius 65 s, Emerald 77 s, Garnet 79 s) | CPU 19–33 s, queue under 1 s |
| IQM VLQ | ~2 min (118 s) | CPU 71 s, queue 17 s |
| AQT PIAST-Q | ~5 h (17,903 s = 4 h 58 min; Slurm, PCSS cluster) | CPU ~9 s, queue 15,503 s, QPU ~2,478 s (~41 min) estimated — AQT reports no QPU time, so the parts (17,990 s) can exceed the measured total |

#### Running AQT PIAST-Q through Slurm on the PCSS Quantum UI cluster

The AQT script is synchronous: it sits in one process, waiting on PIAST-Q after every execution, so it must stay
up for the whole run (~41 min of QPU time estimated for the mini benchmark, 5 h wallclock measured on the PCSS cluster). It can be started
directly on a local computer (see \* above), but that is 1.5–2.5 times slower per circuit. On the PCSS cluster, run it through Slurm; details are in
`source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/readme_aqt.md` §4.3.

1. Connect to PCSS through OpenVPN, then to the login node `quantum-ui` over SSH (PCSS provides the
   instructions), and clone this repository there to `~/Quantum_Benchmarking_project` (the sbatch files
   expect it there; if you clone elsewhere, edit the `PROJECT` line of each of them).
2. Once on a worker node (`srun --pty /bin/bash`), install [uv](https://astral.sh/uv)
   (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and create the two Linux venvs from the project root,
   then `exit` the session:

   ```bash
   uv venv --python 3.12 venvs/local_sim.venv
   uv pip install --python venvs/local_sim.venv -r venvs/requirements_local_sim.txt
   uv venv --python 3.12 venvs/aqt.venv
   uv pip install --python venvs/aqt.venv -r source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/requirements_aqt.txt
   ```

3. Put the token in `tokens/` on the cluster and make it private:
   `chmod 600 tokens/aqt_piast-q_token`.
4. From the login node, build the circuits (once — the cluster has no copy of the circuit cache) and
   start the run. `run_noiseless_all_to_all.sbatch` passes its arguments to
   `benchmark_noiseless_all_to_all.py`, so it builds the cache of whichever benchmark you name (§5.1);
   wait for it to finish (`squeue -u $USER`) before starting the loop:

   ```bash
   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_noiseless_all_to_all.sbatch mini           # builds the mini circuits (use `full` for the full ones)
   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_health_check_aqt.sbatch backend=PIAST-Q    # optional: one health check
   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop.sbatch backend=PIAST-Q mini       # the mini benchmark (`full`: the full one); random layout without a calibration file
   squeue -u $USER                                       # its state; scancel <jobid> stops it
   ```

   Each job writes its Slurm log `<job-name>_<jobid>.out` into the project root, and the loop script's own log
   goes to `output.log` in the run folder.

`run_aqt_loop.sbatch` has a 24 h limit per job, so it pauses itself after 10 h at the end of a chunk and
requeues itself; it resumes where it stopped (the mini benchmark finishes in one go). To also rerun it hourly after a failure
(for up to 7 days), use `run_aqt_loop_with_retries.sbatch` in the same folder with the same arguments. To stop a job after its current chunk, `touch datafiles/<run folder>/STOP` (on Windows: `New-Item datafiles\<run folder>\STOP`; `readme_aqt.md` §4.3). Analyze the run afterwards as in
§5.3 — copy `datafiles/<run folder>` to the machine you analyze on.

IBM and IQM submission returns immediately; run the
collect script whenever you like afterwards (even in a new session) — it collects the jobs that have
finished without waiting for the others, merges them and analyzes the run (run it again later for the rest). Every script is safe to re-run after an interruption: it resumes where it
stopped.

### 5.3 Analyze (`local_sim.venv`)

```powershell
python source\analysis_scripts\run_post_process.py mini --include-real-data
```

Plots land in `analysis/plots+summaries/<run_folder>/`, the summary (timings, cost estimate) in
`datafiles/<run_folder>/processed_results/summary.json`. A run already analyzed, with no chunks collected since, is skipped
with a warning; delete its `datafiles/<run_folder>/processed_results/` to analyze it anew.

Plots comparing all the runs — AWAE and cumulative AWAE of each run's best EMS vs $N_t$,
QPU time and effective cost per QPU — land in `analysis/plots+summaries/mini_collective_real-data/` (the folder name says which kinds of runs it compares). Add
`--include-simulations-real` / `--include-simulations-noiseless` to compare the QPUs with
simulations too, `--date=FROM:TO`, `--backends=…`, `--services=…` or `--layouts=…` to analyze only some runs (USER_MANUAL §10), or
`--individual-only` / `--collective-only` to make only one kind of plot; e.g.
`python source\analysis_scripts\run_post_process.py mini --include-real-data --collective-only` redraws the comparison without
re-analyzing anything. USER_MANUAL §10 has the details.

---

## 6. Run the full benchmark on every QPU

The full benchmark: **480 chunks (RC samples), 10 depths (Nt = 1, 3, …, 19), 10 shots per circuit**. One
chunk = 60 benchmark circuits + 8 SPAM circuits = 68 circuits.

### 6.1 Build the full circuits (local, `local_sim.venv`, no cost — but hours of CPU time)

Run the scripts in `source/benchmark_scripts/local_simulation_scripts/` once, from the project root (#3 after #2; #3 uses `ibm.venv`, the others `local_sim.venv`), each with the argument `full`
(e.g. `python source\benchmark_scripts\local_simulation_scripts\benchmark_noiseless_lnn.py full`). Each builds a circuit set
(≈ 3.1 GB on disk for all-to-all, ≈ 4.1 GB for LNN; see "Disk space" below) and then runs the full noiseless benchmark on it — AWAE ≈ 0 confirms the
circuits are right before any QPU time is spent.

**Disk space to expect.** A mini circuit cache is only ≈ 5 MB and a mini run folder ≈ 2–3 MB. A full circuit cache is ≈ 3.1 GB
(all-to-all) or ≈ 4.1 GB (LNN), about 30,000 files each. A finished full run folder, with its raw results, transpiled circuits and
processed results, is ≈ 1–2 GB depending on the QPU (≈ 1.9 GB IBM, ≈ 1.8 GB IQM EQE1, ≈ 1.3 GB IQM VLQ, ≈ 1.1 GB IQM Resonance, ≈ 1.8 GB AQT PIAST-Q). Keep several
GB free per topology and per QPU you run.

| # | Script | Needed for | CPU time: build / simulation |
|---|---|---|---|
| 1 | `benchmark_noiseless_all_to_all.py full` | AQT PIAST-Q, IQM VLQ, IQM Resonance Sirius (saved at QPY version 13, which the older Qiskit of AQT and VLQ reads) | ~60 min / ~15 min |
| 2 | `benchmark_noiseless_lnn.py full` | IBM, IQM EQE1, IQM Resonance Emerald / Garnet | ~60 min / ~20 min |
| 3 | *(optional)* `benchmark_fake_fez_lnn.py full` in `ibm.venv` — noisy reference under FakeFez | — | — (needs #2's circuits) / ~4 h (default `scheduling_simulation=true`), ~65 min with `false` |

(The simulation times were measured at 1000 shots per circuit.) Total: about 2½ h the first time (about 3¾ h with the optional FakeFez run at `scheduling_simulation=false`; the default `true` adds about 3 h to the full FakeFez run). Once the circuits are cached they are reused by every later run.

### 6.2 Optional rehearsal without QPU time (local, no cost)

`source\benchmark_scripts\real_qpu_scripts\IBM\qbcs_real_simulate_ibm.py backend=ibm_marrakesh full`
(`ibm.venv`) and `...\IQM_RESONANCE\qbcs_real_simulate_iqm_resonance.py backend=emerald full` (`iqm_resonance.venv`) download the
device's current calibration and simulate the whole run locally with it: ~1 h CPU time each
(≈ 8 s per chunk with `scheduling_simulation=false`). `...\IQM_VLQ\qbcs_real_simulate_iqm_vlq.py` does the same for VLQ from IT4I's public calibration API (no login). IQM EQE1 and AQT report no calibration, so their simulate scripts take it from a file (`calibration_file=`, by default the newest in `calibration_input_files/<provider>/`) and need `layout=` named explicitly (see **Qubit layout** in §5.2 and their readmes).

Every simulate script takes the optional `scheduling_simulation=true|false` (default `true`; add `scheduling_simulation=false` to any command above for the faster run); see the note in §4 and each readme. With `true` a full run takes ≈ 4 h instead of ≈ 1 h.

### 6.3 Submit and collect

> 💰 **COST — this step runs on real hardware.** Expected cost at list prices, health checks included
> (base numbers — jobs, circuits, shots, QPU time — in the table below it):
>
> | QPU | Calculation | Expected cost |
> |---|---|---|
> | IBM | ~120 s billed (+ a few s of health checks) × $96/min ≈ $200 | ≈ €170–180 (free on the Open Plan) |
> | IQM EQE1 | 482 tasks × $0.30 + 326,600 shots × $0.0016 = $667 | ≈ €590 |
> | IQM Resonance (Emerald) | 482 tasks × $0.30 + 326,600 shots × $0.0016 = $667 | ≈ €590 |
> | IQM Resonance Garnet | 482 tasks × $0.30 + 326,600 shots × $0.00145 = $618 | ≈ €540 |
> | IQM VLQ | 482 tasks (1 chunk of 68 circuits per task, + 2 health checks) × $0.30 + 326,600 shots × $0.00064 = $354 | ≈ €310 |
> | IQM Resonance Sirius | 482 tasks × $0.30 + 326,600 shots × $0.00064 (synthetic price, as for VLQ) = $354 | ≈ €310 |
> | AQT PIAST-Q | 33,121 circuits × €0.30 + 374,500 shots × €0.021 | ≈ **€18,000** |

Same scripts and commands as in §5.2 (AQT: `qbcs_real_submit_aqt_loop.py`), with **`full`** in place of `mini` — the collect scripts
too, e.g. `python ...\IBM\qbcs_real_submit_ibm.py backend=ibm_marrakesh full`, later `python ...\IBM\qbcs_real_collect_ibm.py backend=ibm_marrakesh full`. Before
committing to the whole run, you can lower `CHUNKS_TO_SUBMIT` (IBM, IQM) or `CHUNK_STOP` (AQT) to submit a
few chunks first. Expect ~1½–2 h CPU time on your side for loading and transpiling the circuits (measured
`CPU_time_s` of past runs, circuit build included).

| QPU | Jobs | Circuits | Shots | QPU time (past runs) | Wall time (past runs, incl. queue) |
|---|---|---|---|---|---|
| IBM (Fez, Kingston, Marrakesh) | 10 (48 chunks each) + 2 health checks | 32,640 + 2 | 326,400 + 200 | **~2 min** (12 s billed per 48-chunk job) | ~1 h to ~1 day |
| IQM EQE1 | 480 (1 chunk each) + 2 | 32,640 + 2 | 326,400 + 200 | **~8 h** (0.9 s per circuit) | ~9 h |
| IQM Resonance (Emerald) | 480 (1 chunk each) + 2 | 32,640 + 2 | 326,400 + 200 | **~7 min** (~0.85 s billed per job) | ~1 h |
| IQM VLQ | 40 (12 chunks each; 48-chunk jobs failed at IT4I) + 2 | 32,640 + 2 | 326,400 + 200 | ~3½ min estimated (execution ≈ 3 ms per circuit + 0.3 ms per shot) | submission ~4.2 h (~2.7 h uploading + ~1.5 h transpiling), plus queue |
| AQT PIAST-Q | 480 chunks, one after another + 481 health checks (1 PRE + 1 after each chunk) | 32,640 + 481 | 326,400 + 48,100 | **~28 h** (3 s per circuit ≈ 27 h, health checks ~1 h at ~9 s each) | weeks: the first 168 chunks took ~14 days |

For the AQT full run, keep the Windows
Terminal tab open and stop the PC from sleeping (*Settings → System → Power*) for the duration. On the PCSS
cluster, run the loop script through Slurm with `AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop.sbatch` instead, which
pauses and requeues itself between chunks (`readme_aqt.md` §4.3).

For the IQM EQE1 and IQM VLQ full runs, submit with `python ...\IQM_EQE1\qbcs_real_submit_iqm_eqe1_with_retries.py backend=EQE1 full` and `python ...\IQM_VLQ\qbcs_real_submit_iqm_vlq_with_retries.py backend=VLQ full`, and collect with the plain collect scripts as above. These scripts submit the run and retry a failed submission instead of stopping. The VLQ one waits for the last submitted job to finish on the provider before each submission attempt, retries at once after an attempt that submitted a chunk and after 10 min otherwise, and gives up after 48 h without progress (exit code 14); the EQE1 one does not wait, retries a failed submission at once and then every 1 min, and gives up after 20 min without progress (exit code 14). If every chunk is submitted and only the health check after them could not be submitted, it repeats just that health check, at most 3 times (never a new run). Stop it with Ctrl-C and start it again with the same arguments to resume. Keep the session open as described above.

The layout arguments of §5.2 apply to the full run too: IBM, IQM Resonance and IQM VLQ default to `layout=optimized`; IQM EQE1
and AQT PIAST-Q run on a random layout unless you pass `layout=optimized calibration_file=calibration_input_files/<provider>/<file>.json`, `layout=lowest`
or the qubits (`layout=15,14,21` for IQM EQE1; AQT PIAST-Q has ions 0…19, e.g. `layout=15,14,12`). The run folder ends in `_layout-<mode>_<YYYY-MM-DD>_#<N>`.

### 6.4 Analyze (`local_sim.venv`)

```powershell
python source\analysis_scripts\run_post_process.py full --include-real-data
```

It also works on a run that is still being collected: it analyzes the chunks collected so far into
separate files named after the number of chunks (`run_data+nsamples-<N>.json`, `summary+nsamples-<N>.json`, …), leaving the full run's files untouched.

As in §5.3, the plots comparing all runs land in `analysis/plots+summaries/full_collective_real-data/`, and
`--individual-only` / `--collective-only` make only one kind.
