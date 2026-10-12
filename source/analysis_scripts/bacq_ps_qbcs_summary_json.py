"""
Compute BACQ derived quantities from run_data.json and, if the run has SPAM data, run_data_spam.json
too (the best mitigation is picked out of all 12 candidates), and save the final summary in a json file.
"""

import os
import sys
import json
import glob
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

from source.core.common import exit_with_error, rel_path
from source.core.records.run_info import resolve_run
from source.core.records.run_names import parse_run_name, run_summary_name, service_name_of_run_service, IBM_CLOUD, subset_suffix


from source.core.analysis.derived_quantities import compute_derived_quantities, NaNEncoder, round_floats, cumulative_awae, EPSILON
from source.core.analysis.mitigation import BASE_EMS_LIST
from source.core.analysis.run_timing import aggregate_run_timing, derive_date_time_from_data, qpu_workload
from source.core.records.job_timing import TIMING_NOTES, QPU_S_SOURCES, AQT_MAX_SHOTS, QPU_EXEC_S_SOURCES, QPU_EXECUTION_NOTE
from source.core.records.circuit_check import load_run_verification, print_verification, verification_warning
from source.core.records.chunk_scope import completeness, completeness_warning


# ── Config for extracting the runs ────────────────────────────────────────────────────────────────────
RUN_FOLDER    = "old_runs_full/full_real-data_iqm-emerald_lnn-True_service-iqm-resonance_spam-True_layout-legacy_2026-09-07_#1"   # standalone: the run (datafiles/<RUN_FOLDER>)

N_SAMPLES     = None # None = full run (unchanged behaviour); int = read that
                       # subset's run_data+nsamples-N.json instead and derive N_acc/AWAE
                       # from it — CPU/QPU/queue time and dates below are scoped to
                       # that same subset (see aggregate_run_timing/derive_date_time_from_data)

# ── Optional overrides for source/analysis_scripts/run_post_process.py — standalone behaviour above is unchanged ──
if "QPU_N_SAMPLES" in os.environ:
    _v = os.environ["QPU_N_SAMPLES"]
    N_SAMPLES = None if _v in ("", "None") else int(_v)
RUN_FOLDER, INFO = resolve_run(RUN_FOLDER)   # QPU_RUN_FOLDER wins; the run's own settings win over the constants above
NAME         = parse_run_name(RUN_FOLDER)
BACKEND_NAME = NAME.device          # ibm-fez, iqm-emerald, iqm-vlq, iqm-eqe1, aqt-piast-q, local-aer
IS_SIM       = NAME.is_simulation
NSHOTS, LNN, DO_SPAM = INFO.nshots, INFO.lnn, INFO.do_spam
USE_SPAM = DO_SPAM   # run_data_spam*.json if the run has SPAM data, else run_data*.json
SUFFIX     = subset_suffix(N_SAMPLES)

if IS_SIM and not glob.glob(os.path.join("datafiles", RUN_FOLDER, "job_records",
                                                                  "job_record_chunk_*.json")):
    # simulations made before local job records existed have no job_records/ to derive the
    # simulation window from (newer ones do; no simulation has QPU or queue time).
    DATE_START, DATE_END, SUBMISSION_TO_RESULTS, ACTIVE_DURATION = "N/A", "N/A", None, None
else:
    DATE_START, DATE_END, SUBMISSION_TO_RESULTS, ACTIVE_DURATION = derive_date_time_from_data(
        run_folder=RUN_FOLDER, n_samples=N_SAMPLES, service=None if IS_SIM else service_name_of_run_service(NAME.service))

# ── Runtimes: CPU (build + conversion + transpilation + submission), QPU, queue — health checks
# included (source/core/analysis/run_timing.py: aggregate_run_timing, definitions in source/core/records/job_timing.py) ──
TIMING     = aggregate_run_timing(RUN_FOLDER, n_samples=N_SAMPLES, total_chunks=INFO.sample_size,
                                  qiskit_v13_qpy=INFO.circuits_qiskit_v13_qpy)
CPU_TIME   = TIMING["cpu"]["total_s"]
# The headline QPU, queue and collection times count the failed jobs too (they ran and were billed), so the
# time fractions describe the whole run; the successful jobs' times stay in timing_breakdown, the failed
# jobs' in failed_attempts.
INCL_FAILED = None if IS_SIM else TIMING["failed_attempts"]["totals_incl_failed"]
QPU_TIME   = 0.0 if IS_SIM else INCL_FAILED["qpu_s"]
QUEUE_TIME = 0.0 if IS_SIM else INCL_FAILED["queue_s"]
# QPU execution only — "N/A" when the provider does not report it for every job (and for simulations)
QPU_EXEC   = "N/A" if IS_SIM or TIMING["qpu_execution"]["total_s"] is None \
             else TIMING["qpu_execution"]["total_s"]
# Result collection — None ("N/A", 0 in the total) for runs collected before it was timed
COLLECT_TIME = TIMING["collect"]["total_s"] if IS_SIM else INCL_FAILED["collect_s"]
SERVICE    = None if IS_SIM else service_name_of_run_service(NAME.service)

# Legacy IBM runs recorded no billed time, only execution windows (up to ~4,500 s, far above what IBM bills).
# They get 12 s per started block of 48 chunks instead (480 chunks: 120 s, 49 chunks: 24 s) — for QPU_time_s
# and everything derived from it, and for the cost below. timing_breakdown keeps the recorded windows.
WORKLOAD = None if IS_SIM else qpu_workload(os.path.join("datafiles", RUN_FOLDER), N_SAMPLES)
LEGACY_IBM_S_PER_BLOCK, LEGACY_IBM_CHUNKS_PER_BLOCK = 12.0, 48
if SERVICE == IBM_CLOUD and TIMING["legacy_run"]:
    QPU_TIME = LEGACY_IBM_S_PER_BLOCK * -(-WORKLOAD["n_chunks"] // LEGACY_IBM_CHUNKS_PER_BLOCK)

# ── Pricing (list prices, excluding VAT) — looked up 2026-09-27 ─────────────────
EXCHANGE_RATE_EUR_PER_USD = 0.8778    # 1 / 1.1392, mid-market EUR/USD rate on 2026-09-27 (wise.com)
IBM_PAYG_USD_PER_MIN      = 96.0      # IBM Quantum Pay-As-You-Go, per billed QPU minute
IBM_PRICING_URL           = "https://www.ibm.com/quantum/pricing"
AWS_TASK_USD              = 0.30      # Amazon Braket, per task; a program set (up to 100 circuits) is one task
AWS_MAX_CIRCUITS_PER_TASK = 100
AWS_SHOT_USD              = {"Emerald": 0.00160, "Garnet": 0.00145}
# IQM Resonance credits per QPU second (resonance.iqm.tech, 2026-09-28). VLQ is a Star 24 machine like
# Sirius, which Braket does not offer: its per-shot price is Emerald's scaled by Sirius/Emerald credits.
RESONANCE_CREDITS_PER_S   = {"Emerald": 0.75, "Garnet": 0.5, "Sirius": 0.3}
RESONANCE_PRICING_URL     = "https://resonance.iqm.tech/"
VLQ_SHOT_USD              = AWS_SHOT_USD["Emerald"] * RESONANCE_CREDITS_PER_S["Sirius"] / RESONANCE_CREDITS_PER_S["Emerald"]
AWS_PRICING_URL           = "https://aws.amazon.com/braket/pricing/"
AWS_PROGRAM_SETS_URL      = "https://aws.amazon.com/about-aws/whats-new/2025/08/amazon-braket-program-sets"
SCALEWAY_AQT_CIRCUIT_EUR  = 0.30      # Scaleway QaaS, AQT IBEX-Q1 (QPU-IBEX-12PQ)
SCALEWAY_AQT_SHOT_EUR     = 0.021
SCALEWAY_PRICING_URL      = "https://www.scaleway.com/en/quantum-as-a-service/"

# ── QPU metadata ──────────────────────────────────────────────────────────────
# PRICING picks how the effective cost is computed (below) from this run's own records;
# FREE_BECAUSE says why the actual cost is 0.
PROTOCOLS = "Protocols: bare, NEC, RC, NEC+RC, cRC, NEC+cRC, along with readout error mitigation for each."
if IS_SIM:
    # Not a real QPU — no provider/service/cost to report, N/A placeholders throughout.
    QPU_PROVIDER  = "N/A"
    ARCHITECTURE  = "N/A"
    DEVICE_LABEL = f"{NAME.figure_label}, lnn-{NAME.lnn}"   # short label for the collective plots
    QPU_SERVICE   = {"sim-noiseless": "Local noiseless simulator",
                     "sim-coherent":  "Local simulator (coherent CNOT noise)"}.get(NAME.data_type, "Local simulator")
    PRICING       = None
    FREE_BECAUSE  = None

elif BACKEND_NAME in ("ibm-marrakesh", "ibm-kingston", "ibm-fez"):
    QPU_PROVIDER  = "IBM"
    ARCHITECTURE  = "Heron R2"
    QPU_SERVICE   = "IBM Quantum Platform (cloud)"
    PRICING       = "ibm_payg"
    FREE_BECAUSE  = "the benchmark fits within the free IBM allocation of the IBM Quantum Open Plan"

elif BACKEND_NAME == "iqm-eqe1":
    QPU_PROVIDER  = "IQM"
    ARCHITECTURE  = "Crystal"
    QPU_SERVICE   = "Euro HPC, Leibniz Supercomputing Centre"
    PRICING       = ("aws", "Emerald")      # EQE1 is the same IQM model as Emerald (54 qubits)
    FREE_BECAUSE  = "of the free access to the machine within the EuroHPC project"

elif BACKEND_NAME == "iqm-emerald":
    QPU_PROVIDER  = "IQM"
    ARCHITECTURE  = "Crystal 54"
    QPU_SERVICE   = "IQM Resonance (cloud)"
    PRICING       = ("aws", "Emerald")
    FREE_BECAUSE  = "of the IQM Resonance credits granted by IQM"

elif BACKEND_NAME == "iqm-garnet":
    QPU_PROVIDER  = "IQM"
    ARCHITECTURE  = "Crystal 20"
    QPU_SERVICE   = "IQM Resonance (cloud)"
    PRICING       = ("aws", "Garnet")
    FREE_BECAUSE  = "of the IQM Resonance credits granted by IQM"

elif BACKEND_NAME == "iqm-sirius":
    QPU_PROVIDER  = "IQM"
    ARCHITECTURE  = "Star 24"
    QPU_SERVICE   = "IQM Resonance (cloud)"
    PRICING       = "aws_vlq"               # Braket does not offer Sirius: same synthetic Star 24 estimate as VLQ
    FREE_BECAUSE  = "of the IQM Resonance credits granted by IQM"

elif BACKEND_NAME == "iqm-vlq":
    QPU_PROVIDER  = "IQM"
    ARCHITECTURE  = "Star 24"
    QPU_SERVICE   = "EuroHPC, IT4Innovations (VLQ, via LEXIS/HEAppE)"
    PRICING       = "aws_vlq"
    FREE_BECAUSE  = "of the free access to the machine within the EuroHPC project"

elif BACKEND_NAME == "aqt-piast-q":
    QPU_PROVIDER  = "AQT"
    ARCHITECTURE  = "Trapped ion (20 qubits)"          # AQT IBEX-class system
    QPU_SERVICE   = "EuroHPC, PSNC (Poznań Supercomputing and Networking Center)"
    PRICING       = "scaleway_aqt"
    FREE_BECAUSE  = "of the free access to the machine within the EuroHPC project"

else:
    exit_with_error(f"{RUN_FOLDER}: no QPU metadata for {BACKEND_NAME} — add it to bacq_ps_qbcs_summary_json.py")
if not IS_SIM:
    DEVICE_LABEL = f"{NAME.figure_label}, lnn-{NAME.lnn}"   # same short label as a simulation (run_names.py)
CONNECTIVITY = "Linear nearest neighbor" if LNN else "All to all"
EXTRA_INFO   = PROTOCOLS

if IS_SIM:
    EXTRA_INFO = PROTOCOLS + " Local simulation — no QPU cost."

    # sim-qpu-noise runs (source/benchmark_scripts/real_qpu_scripts/*/qbcs_real_simulate_*.py): noisy, from a saved calibration
    _snapshot_path = os.path.join("datafiles", RUN_FOLDER, "calibration_snapshot.json")
    if os.path.exists(_snapshot_path):
        with open(_snapshot_path, encoding="utf-8") as f:
            _snap = json.load(f)
        _cal = f", calibration set {_snap['calibration_set_id']}" if _snap.get("calibration_set_id") else ""
        QPU_SERVICE = (f"Local Aer simulation with the calibration of {_snap['backend']} "
                       f"downloaded {_snap['downloaded_at']}{_cal}")
        EXTRA_INFO  = (EXTRA_INFO + f" Noisy simulation from a calibration snapshot of {_snap['backend']} "
                       f"({_snap['downloaded_at']}) — not a hardware result.")

# ── Cost: what this run would cost at list prices; actually paid: 0 ───────────
# Priced from the run's own records (the analysed subset, health checks included): source/core/analysis/run_timing.py: qpu_workload.
ACTUAL_COST_EUR = 0
EFFECTIVE_COST_USD = EXCHANGE_RATE = None      # written only for USD-priced backends
NO_VAT = " The effective cost excludes VAT."
if PRICING is None:
    EFFECTIVE_COST_EUR = 0
    COST_EXPLANATION   = "Local simulation — no QPU cost."
else:
    n_jobs, n_circ, n_shots = WORKLOAD["n_jobs"], WORKLOAD["n_circuits"], WORKLOAD["n_shots"]
    counted = (f"{n_jobs:,} jobs ({WORKLOAD['n_chunk_jobs']:,} for the chunks, {WORKLOAD['n_health_checks']} "
               f"health checks), {n_circ:,} circuits, {n_shots:,} shots")
    if WORKLOAD.get("n_split_chunks"):
        counted += (f" ({WORKLOAD['n_split_chunks']:,} chunks were run as several executions of at most "
                    f"{AQT_MAX_SHOTS} shots per circuit, AQT's limit; each execution counts as circuits)")
    n_failed_jobs, n_failed_circ, n_failed_shots = (WORKLOAD["n_failed_jobs"], WORKLOAD["n_failed_circuits"],
                                                    WORKLOAD["n_failed_shots"])
    if n_failed_jobs or n_failed_circ:      # failed jobs that reached the provider (lost results) are billed too
        counted += (f"; plus {n_failed_jobs:,} failed jobs, {n_failed_circ:,} circuits, {n_failed_shots:,} shots "
                    f"from failed attempts (their results were lost), included in the cost")
    if WORKLOAD["n_may_exist_on_provider"]:
        counted += (f"; {WORKLOAD['n_may_exist_on_provider']:,} submissions may have run on the provider "
                    f"(cost unknown, not counted)")
    if WORKLOAD["n_uncollected"]:
        counted += (f"; {WORKLOAD['n_uncollected']:,} jobs were submitted but never collected "
                    f"(cost unknown, not counted)")
    if PRICING == "ibm_payg":
        EFFECTIVE_COST_USD = QPU_TIME * IBM_PAYG_USD_PER_MIN / 60
        model = (f"Paying the QPU time ({QPU_TIME:,.0f} s"
                 + (f", of which {TIMING['failed_attempts']['qpu_s']:,.0f} s in failed jobs"
                    if TIMING["failed_attempts"]["qpu_s"] and not TIMING["legacy_run"] else "")
                 + (f"; this legacy run recorded no billed time, so it is set to {LEGACY_IBM_S_PER_BLOCK:.0f} s per "
                    f"started block of {LEGACY_IBM_CHUNKS_PER_BLOCK} chunks, for {WORKLOAD['n_chunks']} chunks"
                    if TIMING["legacy_run"] else "")
                 + f") at the IBM Quantum Pay-As-You-Go price of ${IBM_PAYG_USD_PER_MIN:.0f} per QPU minute "
                   f"({IBM_PRICING_URL})")
    elif PRICING == "scaleway_aqt":
        EFFECTIVE_COST_EUR = (n_circ + n_failed_circ) * SCALEWAY_AQT_CIRCUIT_EUR + (n_shots + n_failed_shots) * SCALEWAY_AQT_SHOT_EUR
        model = (f"Paying at Scaleway's price for the AQT IBEX-Q1 QPU, €{SCALEWAY_AQT_CIRCUIT_EUR:.2f} per circuit + "
                 f"€{SCALEWAY_AQT_SHOT_EUR} per shot ({SCALEWAY_PRICING_URL}), for this run's {counted}")
    elif PRICING == "aws_vlq":                         # synthetic: Braket's scheme, Emerald's shot price scaled
        # A VLQ job holds up to 12 chunks (VLQ_CHUNKS_PER_JOB), more than one program set: it counts as the tasks its whole
        # chunks would fill (floor(100 / circuits per chunk) chunks per task); a health check is one task.
        per_task = AWS_MAX_CIRCUITS_PER_TASK // WORKLOAD["circuits_per_chunk"]
        # The failed jobs count the same way: their chunks, and each failed health check as one task.
        n_failed_hc = n_failed_jobs - len(WORKLOAD["failed_chunks_per_job"])
        n_tasks  = (sum(-(-k // per_task) for k in WORKLOAD["chunks_per_job"] + WORKLOAD["failed_chunks_per_job"])
                    + WORKLOAD["n_health_checks"] + n_failed_hc)
        EFFECTIVE_COST_USD = n_tasks * AWS_TASK_USD + (n_shots + n_failed_shots) * VLQ_SHOT_USD
        model = (f"A synthetic estimate combining the IQM Resonance and Amazon Braket billing schemes (VLQ is not "
                 f"sold commercially): Braket's ${AWS_TASK_USD:.2f} per task, and a per-shot price of "
                 f"${VLQ_SHOT_USD:.5f} — Braket's IQM Emerald price (${AWS_SHOT_USD['Emerald']}, {AWS_PRICING_URL}) "
                 f"scaled by IQM Resonance's credits per QPU second for Sirius, a Star 24 machine like VLQ "
                 f"({RESONANCE_CREDITS_PER_S['Sirius']}), over Emerald's ({RESONANCE_CREDITS_PER_S['Emerald']}) "
                 f"({RESONANCE_PRICING_URL}). Each job counts as the program sets of up to "
                 f"{AWS_MAX_CIRCUITS_PER_TASK} circuits its whole chunks fill ({per_task} chunks of "
                 f"{WORKLOAD['circuits_per_chunk']} circuits each), each billed as one task ({AWS_PROGRAM_SETS_URL}), "
                 f"and each health check as one task: {n_tasks:,} tasks for this run's {counted}")
    else:                                              # ("aws", machine): one IQM job = one Braket task (a program set)
        machine = PRICING[1]
        assert WORKLOAD["max_circuits_per_job"] <= AWS_MAX_CIRCUITS_PER_TASK, (
            f"a job has {WORKLOAD['max_circuits_per_job']} circuits, more than an AWS program set holds "
            f"({AWS_MAX_CIRCUITS_PER_TASK}) — the one-job-one-task pricing does not apply")
        EFFECTIVE_COST_USD = (n_jobs + n_failed_jobs) * AWS_TASK_USD + (n_shots + n_failed_shots) * AWS_SHOT_USD[machine]
        model = (f"Paying at Amazon Braket's price for IQM {machine}, ${AWS_TASK_USD:.2f} per task + "
                 f"${AWS_SHOT_USD[machine]} per shot ({AWS_PRICING_URL}), with each job submitted as one program "
                 f"set of up to {AWS_MAX_CIRCUITS_PER_TASK} circuits, which is billed as one task "
                 f"({AWS_PROGRAM_SETS_URL}), for this run's {counted}")
    if EFFECTIVE_COST_USD is not None:
        EXCHANGE_RATE      = EXCHANGE_RATE_EUR_PER_USD
        EFFECTIVE_COST_EUR = EFFECTIVE_COST_USD * EXCHANGE_RATE
        result = (f"would cost ${EFFECTIVE_COST_USD:,.2f}, which is €{EFFECTIVE_COST_EUR:,.2f} at "
                  f"{EXCHANGE_RATE} EUR per USD (mid-market rate on 2026-09-27)")
    else:
        result = f"would cost €{EFFECTIVE_COST_EUR:,.2f}"
    COST_EXPLANATION = f"The actual cost is 0 because {FREE_BECAUSE}. {model} {result}.{NO_VAT}"
COST_FIELDS = {
    **({"effective_cost_USD": round(EFFECTIVE_COST_USD, 2)} if EFFECTIVE_COST_USD is not None else {}),
    **({"exchange_rate_EUR_per_USD": EXCHANGE_RATE} if EXCHANGE_RATE is not None else {}),
    "effective_cost_EUR": round(EFFECTIVE_COST_EUR, 2),
    "actual_cost_EUR":    ACTUAL_COST_EUR,
    "cost_explanation":   COST_EXPLANATION,
}

# ── Load run_data ─────────────────────────────────────────────────────────────
# Whenever SPAM-corrected data was collected for this run (run_data_spam*.json
# exists), load BOTH the no-SPAM and SPAM-corrected files so the derived
# quantities below can pick the best mitigation out of all 12 candidates
# (6 base schemes x with/without SPAM), not just the 6 from USE_SPAM's file.
fname       = f"run_data_spam{SUFFIX}.json" if USE_SPAM else f"run_data{SUFFIX}.json"
path        = os.path.join("datafiles", RUN_FOLDER, "processed_results", fname)
with open(path, encoding="utf-8") as f:
    run_data = json.load(f)

spam_path   = os.path.join("datafiles", RUN_FOLDER, "processed_results", f"run_data_spam{SUFFIX}.json")
nospam_path = os.path.join("datafiles", RUN_FOLDER, "processed_results", f"run_data{SUFFIX}.json")
has_spam_data = os.path.exists(spam_path) and os.path.exists(nospam_path)

COMPLETENESS = run_data.get("completeness") or completeness(os.path.join("datafiles", RUN_FOLDER), INFO.sample_size, N_SAMPLES)
if completeness_warning(COMPLETENESS):
    print(f"  {completeness_warning(COMPLETENESS)}")

depths   = run_data["depths"]
# Derived from the actual run's depths, not the standard
# 10-depth real-QPU sweep — a run with a
# different LIST_TIMES (e.g. a reduced local-sim test) would otherwise raise
# KeyError on depths[str(nt)] for any Nt this run never built.
nt_list      = sorted(int(k) for k in depths.keys())
base_ems_list = BASE_EMS_LIST

# ── Extract AWAE per depth ────────────────────────────────────────────────────
# ems_list: 6 base schemes, plus their "<scheme>+SPAM" counterparts whenever
# both the no-SPAM and SPAM-corrected run_data files are available for this run.
if has_spam_data:
    ems_list = base_ems_list + [f"{ems}+SPAM" for ems in base_ems_list]
    with open(nospam_path, encoding="utf-8") as f:
        depths_nospam = json.load(f)["depths"]
    with open(spam_path, encoding="utf-8") as f:
        depths_spam = json.load(f)["depths"]
    ems_source = {ems: (depths_nospam, ems) for ems in base_ems_list}
    ems_source.update({f"{ems}+SPAM": (depths_spam, ems) for ems in base_ems_list})
else:
    ems_list = list(base_ems_list)
    ems_source = {ems: (depths, ems) for ems in base_ems_list}

awae_Nt       = {ems: [] for ems in ems_list}
awae_cumul_Nt = {ems: [] for ems in ems_list}
Z_Nt          = []

for nt in nt_list:
    Z_Nt.append(depths[str(nt)]["Z_exact"])
    for ems in ems_list:
        src_depths, src_key = ems_source[ems]
        awae_Nt[ems].append(src_depths[str(nt)][src_key]["AWAE"])

Z_Nt = np.array(Z_Nt)
for ems in ems_list:
    awae_Nt[ems] = np.array(awae_Nt[ems])
    awae_cumul_Nt[ems] = list(cumulative_awae(Z_Nt, awae_Nt[ems]))

# ── Were the executed circuits the submitted ones? ────────────────────────────
VERIFICATION = run_data.get("circuit_verification") or load_run_verification(os.path.join("datafiles", RUN_FOLDER), SUFFIX)
print_verification(VERIFICATION)
if verification_warning(VERIFICATION):
    print(f"  ⚠️  {verification_warning(VERIFICATION)}")


# ── Compute derived quantities ────────────────────────────────────────────────
derived = compute_derived_quantities(
    awae_cumul_Nt     = awae_cumul_Nt,
    awae_Nt           = awae_Nt,
    T_cpu_total       = CPU_TIME,
    T_qpu_total       = QPU_TIME,
    T_queue_total     = QUEUE_TIME,
    T_collect_total   = COLLECT_TIME,
    epsilon           = EPSILON,
    cnots_per_trotter = 6,
    Nt_vals           = nt_list,
    analyze_only_bare = False,
    lnn               = LNN
)

# ── Build summary ─────────────────────────────────────────────────────────────
summary = {
    "QPU_provider":          QPU_PROVIDER,
    "Architecture":          ARCHITECTURE,
    "device_label":          DEVICE_LABEL,
    "QPU_service":           QPU_SERVICE,
    "connectivity":          CONNECTIVITY,
    "date_qc_start":         DATE_START,
    "date_qc_end":           DATE_END,
    "QPU_submission_to_results_duration_s": SUBMISSION_TO_RESULTS,   # date_qc_end - date_qc_start (to the second)
    "QPU_submission_to_results_duration_without_dead_time_s": None if ACTIVE_DURATION is None else round(ACTIVE_DURATION),
    "extra_info":            EXTRA_INFO,
    "shots_per_circuit":     NSHOTS,
    **COST_FIELDS,
    "n_samples_analyzed":    run_data.get("n_samples_analyzed", run_data["nchunks"]),  # older run_data*.json (pre-subset-analysis) lack the new key
    # chunks analyzed, uncollected chunks and health checks (older run_data files: computed now)
    "completeness":          COMPLETENESS,
    **derived,
    "QPU_execution_time_s":  QPU_EXEC,
    "timing_definitions_version": TIMING["timing_definitions_version"],
    "timing_notes":          {
        "CPU_time_s":   "circuit construction (myQLM) + conversion to Qiskit + transpilation (chunks, SPAM, "
                        "health checks) + submission; the circuit build is scaled to the analysed subset. "
                        "Runs from before transpilation/submission were logged separately count each chunk's "
                        "whole transpile+submit interval (AQT: minus its wall time, which is queue + QPU).",
        "QPU_time_s":   TIMING_NOTES.get(SERVICE, "No QPU (local simulation).")
                        + (f" This legacy run recorded no billed time, only execution windows "
                           f"(timing_breakdown.qpu), so its QPU time is set to {LEGACY_IBM_S_PER_BLOCK:.0f} s per "
                           f"started block of {LEGACY_IBM_CHUNKS_PER_BLOCK} chunks ({QPU_TIME:.0f} s)."
                           if SERVICE == IBM_CLOUD and TIMING["legacy_run"] else ""),
        "QPU_execution_time_s": QPU_EXECUTION_NOTE,
        "Queue_time_s": "per job, from the end of submission to the start of execution. IBM, IQM Resonance, "
                        "IQM VLQ and IQM EQE1 queue many jobs at once, so the waits overlap: the longest single wait over all "
                        "chunk jobs and health checks, failed jobs included. Synchronous services (AQT, local) run their jobs one after "
                        "another: the sum of the waits.",
        "QPU_submission_to_results_duration_s": "date_qc_end - date_qc_start: earliest submission to the latest "
                                                "result over the chunk, failed-attempt and health-check jobs",
        "QPU_submission_to_results_duration_without_dead_time_s": "the same window minus the dead time: the gaps "
                                                "between the jobs' merged intervals longer than 30 min (2 h for AQT), "
                                                "e.g. a failure noticed hours later",
        "Collect_time_s": "result collection: from a finished job to its saved counts — downloading the result, "
                          "downloading the executed circuits for the circuit check, saving counts and records — "
                          "for every chunk and health check, failed jobs included; waiting for unfinished jobs is not included. AQT and "
                          "the local simulator return results synchronously: handling them. N/A for runs "
                          "collected before it was timed (then 0 in the total and FT_collect N/A).",
        "FT":           "FT_clas + FT_quant + FT_queue + FT_collect = 1, "
                        "T_generation_submission_execution_s = CPU + QPU + queue + collection",
        "qpu_s_sources": QPU_S_SOURCES,
        "qpu_execution_sources": QPU_EXEC_S_SOURCES,
        "failed_attempts": "job attempts that failed and were resubmitted (job_records/failed_attempts/) or are "
                           "still failed. QPU_time_s, Queue_time_s, Collect_time_s (and so the FT fractions and the cost) "
                           "include the failed jobs that reached the provider; timing_breakdown counts only the "
                           "successful jobs, and failed_attempts gives "
                           "the failed jobs' QPU, queue and collection time and the totals including them. Their "
                           "transpilation and submission are already part of CPU_time_s. n_submission_failures: "
                           "jobs whose submission did not succeed (after retrying errors where the provider was "
                           "never reached); n_may_exist_on_provider: those whose connection broke after the request "
                           "may have arrived — check the provider for such a job.",
    },
    "failed_attempts":       {k: TIMING["failed_attempts"][k] for k in
                              ("n_attempts", "n_jobs", "n_submission_failures", "n_may_exist_on_provider",
                               "chunks", "qpu_s", "queue_s", "collect_s", "totals_incl_failed")},
    "legacy_run":            TIMING["legacy_run"],   # old IBM / IQM records: times from the old fields
    "timing_breakdown":      TIMING,
    "circuit_verification":  VERIFICATION,
}

summary = round_floats(summary)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join("datafiles", RUN_FOLDER, "processed_results", run_summary_name(N_SAMPLES))
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, cls=NaNEncoder)

print(f"✅ Summary saved → {rel_path(out_path)}")
print(f"   N_max={derived['N_max']}, N_acc={derived['N_acc']}, best_ems={derived['best_ems_Nmax']}")

