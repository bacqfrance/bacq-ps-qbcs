"""
testing_functions.py
====================
Helper functions for the bacq_ps_qbcs pipeline test suite.

Provides three reusable functions:
    - do_qiskit_pipeline()  : runs the new bacq_ps_qbcs Qiskit pipeline and returns run_data dict
    - do_myqlm_pipeline()   : runs the legacy run_mitigation_pipeline_all_depths() (defined at the end of this file) and returns run_data dict
    - do_exact_computation() : computes exact noiseless observables per depth using myQLM statevector QPU

All functions return results as dicts compatible with run_data.json format.

Also provides rm_glob() — a cross-platform (incl. Windows) replacement for
the tests' old `os.system("rm -rf <pattern>")` cleanup calls; see its
docstring.

rm_test_caches() deletes the circuit caches the tests build. They all live in
datafiles/tests/circuits/ (TEST_CIRCUITS_ROOT, set by conftest.py), apart from
the caches real runs use.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import glob
import shutil
import stat
import sys
import time
import numpy as np
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from source.core.qbcs import bacq_ps_qbcs
from pathlib import Path
from source.core.circuits.building import (
    build_BCS_time_evolution_circuit, build_postplugin, build_preplugin, calculate_infidelity_theta,
    calculate_thetas_from_infidelity, compute_exact_observables, get_default_qpu, load_obs_and_shots,
    load_or_compile_batch, on_tgcc, prepare_initial_batch)
from source.core.circuits.SPAM_functions import build_spam_calibration_batch, compute_and_save_spam_confusion_matrix, extract_spam_calibration_counts
from source.core.common import get_circuit_folder, get_exp_file, get_exp_folder, get_run_subfolder, load_json, migrate_flat_chunk_files, PROJECT_ROOT, rel_path
from source.core.records.timestamps import timestamp, TIMESTAMP_FILENAME
import gc
import json


def _clear_readonly(func, path, exc):
    """shutil.rmtree error hook — clears Windows' read-only attribute before
    retrying the failed operation. No-op-equivalent on POSIX. Works as either
    the pre-3.12 `onerror` callback (3rd arg = exc_info tuple) or the 3.12+
    `onexc` callback (3rd arg = the exception) — this function ignores that
    argument either way, so the same one serves both."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


# shutil.rmtree's `onerror` kwarg is deprecated since Python 3.12 in favor of
# `onexc` (same callback signature otherwise) — pick whichever this
# interpreter wants so no DeprecationWarning fires on every test run.
_RMTREE_ERROR_KW = "onexc" if sys.version_info >= (3, 12) else "onerror"


def run_name_for_test(label, data_type="sim-noiseless"):
    """A test's run folder (relative to datafiles/), named as core/records/run_names.py says:
    tests/<label>_<data type>_local-aer_lnn-none_service-local_spam-none_layout-none, e.g.
    tests/test07+nshots-250_sim-noiseless_…. Every test writes under datafiles/tests/ and analysis/tests/
    (run_names.plots_dir / summaries_dir), both git-ignored."""
    from source.core.records.run_names import local_run_name, TESTS_FOLDER
    return f"{TESTS_FOLDER}/{local_run_name(label, data_type, None, None)}"


# Every cache the tests build goes here: conftest.py sets BACQ_CIRCUITS_ROOT to it for the session.
TEST_CIRCUITS_ROOT = "datafiles/tests/circuits"


def rm_test_caches():
    """Delete every circuit cache the tests built (datafiles/tests/circuits/*). The caches
    real runs use are elsewhere (datafiles/circuits/<topology>_Nt<steps>/) and never touched."""
    rm_glob(os.path.join(TEST_CIRCUITS_ROOT, "*"))


def rm_glob(*patterns, retries=5, delay=0.2):
    """
    Cross-platform (incl. Windows, where there is no `rm` binary at all)
    replacement for `os.system("rm -rf <pattern>")`. Expands each glob
    pattern and removes every match — a directory via shutil.rmtree, a file
    via os.remove — silently doing nothing if a pattern matches nothing
    (same as `rm -rf`). Accepts several patterns in one call, matching call
    sites that used to pass more than one to a single `rm -rf`.

    Retries briefly on a transient PermissionError/OSError: Windows can hold
    a just-closed file handle locked a beat longer than POSIX does, and
    clears the read-only attribute Windows sometimes sets before retrying a
    failed delete.
    """
    for pattern in patterns:
        for path in glob.glob(pattern):
            def _remove(path=path):
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, **{_RMTREE_ERROR_KW: _clear_readonly})
                else:
                    os.chmod(path, stat.S_IWRITE)
                    os.remove(path)

            for attempt in range(retries):
                try:
                    _remove()
                    break
                except OSError:
                    if attempt == retries - 1:
                        raise
                    time.sleep(delay)


def do_qiskit_pipeline(
    INFIDELITY    = 0.0,
    RATIO_CT      = 0.5,
    NOISE_TYPE    = "pauli",
    SAMPLE_SIZE   = 16,
    LIST_TIMES    = [0.2, 0.6, 1.4, 1.8],
    DT            = 0.2,
    LNN           = False,
    testing_folder= None,
    nshots        = int(1e4),
    do_SPAM       = False,
    EMS_LIST      = None,
    sampler       = None
    ):

    testing_folder = testing_folder or run_name_for_test("test01")
    if EMS_LIST is None:
        EMS_LIST = ["NEC", "RC+NEC", "cRC+NEC"] if not do_SPAM else ["SPAM", "NEC", "RC+NEC", "cRC+NEC"] 
    chunk_size = int(len(LIST_TIMES) * len([e for e in EMS_LIST if e != "SPAM"]) * 2)
    # ── Step 1: Run new Qiskit pipeline ───────────────────────────────────────────
    print("\n=== Step 1: New Qiskit pipeline ===")
    qbcs = bacq_ps_qbcs(
        infidelity_total            = INFIDELITY,
        ratio_ct                    = RATIO_CT,
        noise_type                  = NOISE_TYPE,
        sample_size                 = SAMPLE_SIZE,
        chunk_size                  = chunk_size,
        list_circuit_times          = LIST_TIMES,
        error_mitigation_strategies = EMS_LIST,
        do_SPAM                     = do_SPAM,
        nshots                      = nshots,
        dt                          = DT,
        folder_path                 = testing_folder,
    )

    qbcs.build(circuit_transpilation_LNN=LNN)
    sampler = AerSamplerV2() if sampler is None else sampler
    qbcs.submit(
        run_folder = testing_folder,
        lang       = "qiskit",
        service    = "local",
        backend    = sampler,
    )
    qbcs.analyze_results()


    # Load Qiskit results
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    run_data_path = os.path.join(base_dir, "datafiles", testing_folder, "processed_results", "run_data.json")
    with open(run_data_path, encoding="utf-8") as f:
        qiskit_results = json.load(f)

    return qiskit_results


def do_myqlm_pipeline(
    INFIDELITY    = 0.0,
    RATIO_CT      = 0.5,
    NOISE_TYPE    = "pauli",
    SAMPLE_SIZE   = 16,
    LIST_TIMES    = [0.2, 0.6, 1.4, 1.8],
    DT            = 0.2,
    LNN           = False,
    testing_folder= "tests/test01_myqlm",
    nshots        = int(1e4),
    do_SPAM       = False,
    EMS_LIST      = ["NEC", "RC+NEC", "cRC+NEC"]
    ):

    # ── Step 2: Run old myQLM pipeline ────────────────────────────────────────────
    print("\n=== Step 2: Old myQLM pipeline ===")
    run_mitigation_pipeline_all_depths(
        infidelity_total        = INFIDELITY,
        ratio_ct                = RATIO_CT,
        noise_type              = NOISE_TYPE,
        sample_size             = SAMPLE_SIZE,
        list_circuit_times      = LIST_TIMES,
        folder_path             = testing_folder,
        nshots                  = nshots,
        infidelity_from_counts  = True,
        nec_random_initial_state= True,
        do_SPAM                 = do_SPAM,
        LNN                     = LNN,
        error_mitigation_strategies = EMS_LIST
    )

    # Load myqlm results
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    run_data_path = os.path.join(base_dir, "datafiles", testing_folder, "processed_results", "run_data.json")
    with open(run_data_path, encoding="utf-8") as f:
        myqlm_results = json.load(f)

    return myqlm_results

def do_exact_computation(
    LIST_TIMES    = [0.2, 0.6, 1.4, 1.8],
    DT            = 0.2,
    LNN           = False
    ):

    # ── Step 3: Compute exact values ──────────────────────────────────────────────
    print("\n=== Step 3: Exact values ===")
    exact = {}
    for T in LIST_TIMES:
        nt = int(round(T / DT))
        circuit_original, _ = build_BCS_time_evolution_circuit(
            nsites=3, g=0.5, epsilon=np.array([-1, 0, 1]),
            T=T, dt=DT, local_qubit_basis=["X","Y","Z"], LNN=LNN
        )
        exact_vals, _, _ = compute_exact_observables(circuit_original, return_jobs=True)
        exact[nt] = np.real(exact_vals)
        print(f"  Nt={nt}: exact = {np.round(exact[nt], 4)}")

    return exact


# ── The old all-in-one myQLM driver the reference test (do_myqlm_pipeline) runs ─────────────────

def run_mitigation_pipeline_all_depths(infidelity_total, ratio_ct, noise_type="depolarizing", 
                                       folder_prefix="results", nshots=0, sample_size=480, 
                                       list_circuit_times=None, 
                                       infidelity_from_counts=False, nec_random_initial_state=False,
                                       do_SPAM=False, SPAM_shots_per_state=10, 
                                       folder_path=None, LNN=False,
                                       error_mitigation_strategies = ["NEC", "RC+NEC", "cRC+NEC"]):

    from qat.core import Batch
    from source.core.circuits.main import HardwareSubmission_plugin, interleave_the_batch_multidepth_from_disk
    from source.core.analysis.mitigation import merge_chunks, post_processing_and_saving
    noise_ = "dep" if noise_type=="depolarizing" else "pauli"
    if infidelity_total > 1e-8:
        cnot_noise_angle, t2, neighbor_noise_angle = calculate_thetas_from_infidelity(I_tot=infidelity_total, r_ct=ratio_ct, noise=noise_)
    else:
        cnot_noise_angle=0.0; t2=0.0; neighbor_noise_angle=0.0
    if abs(cnot_noise_angle - neighbor_noise_angle) < 1e-15:
        neighbor_noise_angle = neighbor_noise_angle * (1 + 1e-10)
    # Recompute Itest with perturbed angle
    Itest = calculate_infidelity_theta(theta_1=cnot_noise_angle, theta_2=t2, theta_3=neighbor_noise_angle, noise=noise_)
    assert abs(infidelity_total - Itest) < 1e-6  # loosened from 1e-12

    print(f"  [Build] Noise angles: CNOT {cnot_noise_angle}, T2 {t2}, neighbour {neighbor_noise_angle}")

    if list_circuit_times is None: 
        list_circuit_times = list(np.arange(0.2, 4.0, 0.4))

    nec_ems_strategies = [e for e in error_mitigation_strategies if e != "SPAM"]
    ems_to_job_name = {"NEC": "raw", "RC+NEC": "rc", "cRC+NEC": "crc"}
    job_names  = [ems_to_job_name[ems] for ems in nec_ems_strategies]
    roles      = [tag for name in job_names for tag in (name, f"{name}_nec")]
    n_roles    = len(roles)
    chunk_size = len(list_circuit_times) * n_roles

    CONFIG = {
        "noise_val": cnot_noise_angle,
        "neighbor_noise_val": neighbor_noise_angle,
        "nsites": 3,
        "epsilon": np.array([-1, 0, 1]),
        "g": 0.5,
        "T_list": list_circuit_times,
        "ind_T": None, 
        "dt": 0.2,
        "samples": sample_size, #512,
        "split_raw+nec_jobs": True, #CHANGE BEFORE RUNNING
        "nec_measurement_counts": infidelity_from_counts, #CHANGE BEFORE RUNNING
        "nec_randomize_input_state": nec_random_initial_state,
        "chunk_size": chunk_size,
        "obs_str": "XYZ",
        "rng_seed": 0,
        "cnots_per_trotter": 6 if not LNN else None,
        "local_qubit_basis": ["X", "Y", "Z"],
        "shots" : nshots,
        "SPAM_correction": do_SPAM,
        "SPAM_circuit_shots_per_input_state": SPAM_shots_per_state,
        "lnn_architecture" : LNN
    }
    CONFIG["folder_path"]           = os.path.join(PROJECT_ROOT, f"datafiles/runs_I{infidelity_total:.2e}_rct{ratio_ct:.2e}_{noise_type}_samples{CONFIG['samples']}_shots{CONFIG['shots']}_seed{CONFIG['rng_seed']}") if folder_path is None else os.path.join(PROJECT_ROOT, "datafiles", folder_path)
    CONFIG["T"]                     = None if CONFIG["ind_T"] is None else CONFIG["T_list"][CONFIG["ind_T"]]
    CONFIG["exp_label"]             = f"{folder_prefix}_{noise_type}_{CONFIG['noise_val']:.1e}_{CONFIG['neighbor_noise_val']:.1e}"
    CONFIG["exp_label_nec"]         = f"{CONFIG['exp_label']}_nec"
    CONFIG["Nt"]                    = int(round(CONFIG["T"] / CONFIG["dt"])) if CONFIG["T"] is not None else None
    CONFIG["exp_folder"]            = get_exp_folder(config=CONFIG, top_folder=True)
    CONFIG["exp_folder_nec"]        = f"{CONFIG['folder_path']}"#/{CONFIG['exp_label_nec']}_circuit_twirls_ns{CONFIG['samples']}_ntrotter{CONFIG['Nt']}_seed{CONFIG['rng_seed']}"

    if noise_type=="depolarizing":
        CONFIG["noise_angles"]          = np.array([1,1,1,1,1,1]) * CONFIG["noise_val"] # np.ones(6) * CONFIG["noise_val"]
        CONFIG["neighbor_noise_angles"] = np.array([1,1,1]) * CONFIG["neighbor_noise_val"] # np.ones(3) * CONFIG["neighbor_noise_val"]
    elif noise_type=="pauli":
        CONFIG["noise_angles"]          = np.array([1,0,0,1,0,0]) * CONFIG["noise_val"] # np.ones(6) * CONFIG["noise_val"]
        CONFIG["neighbor_noise_angles"] = np.array([1,0,0]) * CONFIG["neighbor_noise_val"] # np.ones(3) * CONFIG["neighbor_noise_val"]


    assert len(CONFIG["epsilon"])==CONFIG["nsites"]
    assert len(CONFIG["local_qubit_basis"])==CONFIG["nsites"]
    
    try:
        post_processing_and_saving(CONFIG=CONFIG)
        print("post_processing_and_saving succeeded on existing results — returning early.")
        return 0
    except Exception as e:
        print(f"post_processing_and_saving failed ({e}) — running full pipeline.")


    TIMESTAMP_FILE = Path(get_exp_file(config=CONFIG, filename=TIMESTAMP_FILENAME, top_folder=True))
    timestamp(label="start", timestamp_file=TIMESTAMP_FILE)

    # Mirrors qbcs.py's build()/interleave_job_batches(): also log the
    # circuit-build boundary events to the shared per-topology circuit-cache
    # log, not just this run's own TIMESTAMP_FILE — circuit_build_breakdown()
    # (used by aggregate_run_timing()) reads build time from there, keyed
    # only by topology, since that cache is shared across every run of it.
    CIRCUIT_TIMESTAMP_FILE = Path(get_circuit_folder(config=CONFIG, top_folder=True)) / TIMESTAMP_FILENAME


    trotter_steps_list = np.round(np.array(CONFIG["T_list"]) / CONFIG["dt"]).astype(int)
    job_tags           = [f"Nt_{num_trotter_steps:02d}_{tag}" for num_trotter_steps in trotter_steps_list for tag in roles]
    n_depths           = len(trotter_steps_list)

    timestamp(label=f"Starting compilation for Nt={trotter_steps_list[0]}.", timestamp_file=CIRCUIT_TIMESTAMP_FILE)

    num_total_jobs = len(CONFIG["T_list"])*(2*len(nec_ems_strategies))*CONFIG["samples"]
    migrate_flat_chunk_files(get_circuit_folder(CONFIG, top_folder=True))
    obs_list         = []
    shots_list       = []
    for ti, t_circuit in enumerate(CONFIG["T_list"]):
        CONFIG["ind_T"]   = ti   
        CONFIG["T"]       = t_circuit
        num_trotter_steps = int(round(t_circuit/CONFIG["dt"]))
        CONFIG["Nt"]      = num_trotter_steps
        timestamp(label=f"Starting compilation for Nt={num_trotter_steps}.", timestamp_file=TIMESTAMP_FILE)
        # ── 1. Build circuits ─────────────────────────────────────────────
        circuit_kwargs = {k: CONFIG[k] for k in ["nsites", "g", "epsilon", "T", "dt", "local_qubit_basis"]}
        circuit_original, rotated_circuit = build_BCS_time_evolution_circuit(**circuit_kwargs, LNN=CONFIG["lnn_architecture"])



        for im, ems in enumerate(nec_ems_strategies):

            nsamples = CONFIG["samples"]

            # ---------------- specifically for RC+NEC ---------------------
            batch_file = os.path.join(
                get_circuit_folder(CONFIG, top_folder=False),
                f"batch_jobs_Nt{CONFIG['Nt']}_{ems}_sample_{nsamples-1}.batch"
            )
            # batch_file   = get_exp_file(config   = CONFIG, filename = f"batch_jobs_{ems}.batch")
            # print("batch_file: ", batch_file)
            # batch_files.append(batch_file)

            # ── 2. Build plugins ──────────────────────────────────────────────
            nec_plugin, rccnot_plugin = build_preplugin(rotated_circuit, CONFIG, error_mitigation=ems, compute_classical_nec_map=(im == 0))

            effective_shots = CONFIG["shots"] # *nsamples if ems=="NEC" else CONFIG["shots"]
            batch = prepare_initial_batch(rotated_circuit, shots=effective_shots, obs=None, job_type=job_names[im])
            print(f"  [Build] {ems}: first compile")
            batch = load_or_compile_batch(nec_plugin,    batch,       None) # Creating the nec circuit and appending it to the original job
            print(f"  [Build] {ems}: second compile")
            batch = load_or_compile_batch(rccnot_plugin, batch, batch_file)
            print(f"  [Build] {ems}: compiled")

            ol, sl = load_obs_and_shots(config=CONFIG, error_mitigation=ems)
            obs_list.extend(ol)
            shots_list.extend(sl)

        print(f"  [Build] {len(obs_list)} observables, {len(shots_list)} shot entries")
        # Merge all jobs from all strategies into a single batch for submission
        #concatenated_batch = Batch(list_of_all_jobs) # 1 Raw noisy circuit + 1 NEC circuit + Ns RC circuits + Ns RC-NEC circuits + Ns cRC circuits + Ns cRC-NEC circuits

        # concatenated_job_list_all_depths.append(list_of_all_jobs)
        # print(f"Length of concatenated jobs at depth {ti}: ", len(concatenated_job_list_all_depths[ti]))
        timestamp(label=f"Finished compilation for Nt={num_trotter_steps}.", timestamp_file=TIMESTAMP_FILE)
    # print(f"Hello: Saved a batch of {len(concatenated_batch):d} jobs")

    # ==================================== Loop over depth over ==============================================

    timestamp(label=f"Starting slicing.", timestamp_file=TIMESTAMP_FILE)
    batch_names = list(nec_ems_strategies)
    chunk_paths = interleave_the_batch_multidepth_from_disk(
        CONFIG, 
        t_list=CONFIG["T_list"], 
        n_rc_samples=CONFIG["samples"],
        batch_names  = batch_names
        )

    

    timestamp(label=f"Finished slicing {num_total_jobs} jobs.", timestamp_file=TIMESTAMP_FILE)
    timestamp(label="Finished interleaving.", timestamp_file=CIRCUIT_TIMESTAMP_FILE)

    # ====================== Compilation is finished and the jobs are sliced across all circuit depths =====================================
    # print("job_map:  ",len(job_map ))
    print(f"  [Build] {len(job_tags)} job tags")
    
    if CONFIG["chunk_size"] is not None:
        chunk_size                  = CONFIG["chunk_size"]
        assert chunk_size%n_depths == 0
        n_chunks                    = int(round(num_total_jobs/chunk_size)) 
        rc_samples_per_ems_chunk    = int(round(chunk_size / (n_depths*2*len(error_mitigation_strategies))))
    else:
        n_chunks                 = 1
        rc_samples_per_ems_chunk = num_total_jobs/(len(job_tags))
        chunk_size               = num_total_jobs
    
    assert abs(n_chunks*chunk_size - num_total_jobs) < 1e-8, \
    "No. of chunks is not consistent with the chunk size and the total batch size!"
    assert chunk_size % len(job_tags) == 0, \
    f"chunk_size ({chunk_size}) not divisible by len(job_tags) ({len(job_tags)})"
    assert rc_samples_per_ems_chunk>0, "no. of samples of each EMS cannot be zero in a chunk!!"
    assert n_chunks == len(chunk_paths)

    print(f"  [Build] {n_chunks} chunks of {chunk_size} jobs, {rc_samples_per_ems_chunk} RC samples per strategy and chunk")

    if os.path.exists(get_exp_file(CONFIG, "results_counts.json", top_folder=True)) and os.path.exists(get_exp_file(CONFIG, "results_fidelity.json", top_folder=True)):
        submit_to_qpu = False
    else:
        submit_to_qpu = True
    
    if submit_to_qpu:
        folder_name = get_exp_folder(config=CONFIG, top_folder=True)
        for nc in range(n_chunks): # Submitting chunk by chunk to the QPU
            if CONFIG["SPAM_correction"]:
                spam_batch = build_spam_calibration_batch(nqubits=CONFIG["nsites"], 
                                                          shots_per_state=CONFIG["SPAM_circuit_shots_per_input_state"])
                qpu = get_default_qpu()
                spam_result_batch = qpu.submit(spam_batch)
                extract_spam_calibration_counts(spam_result_batch, CONFIG["nsites"], 
                                                CONFIG["SPAM_circuit_shots_per_input_state"],
                                                chunk_id=nc, config=CONFIG, 
                                                append_counts_dict=True)

            chunk_batch = Batch.load(chunk_paths[nc])
    
            print(f"Submitting chunk {nc+1}/{n_chunks} — jobs {nc*chunk_size} to {(nc+1)*chunk_size - 1}")
    
            rc_nec_pp = build_postplugin(CONFIG, obs_list, shots_list, error_mitigation="all_split", 
                                        job_tags=job_tags, rc_samples_per_chunk=rc_samples_per_ems_chunk, 
                                        chunk_index=nc, ndepths=len(CONFIG["T_list"]), ems_list=roles,                                       # <-- new
                                        infidelity_from_counts=CONFIG["nec_measurement_counts"])
            
            hs_plugin = HardwareSubmission_plugin(n_rc_samples=CONFIG["samples"], n_shots_per_sample=CONFIG["shots"], 
                                                  folder_name=folder_name, is_the_batch_interleaved=True, 
                                                  chunk_samples_rc=rc_samples_per_ems_chunk, 
                                                  chunk_size=chunk_size, 
                                                  chunk_id=nc, ndepths=len(CONFIG["T_list"]),
                                                  roles=roles)
    
            qpu   = get_default_qpu()
            stack = rc_nec_pp | hs_plugin | qpu
    
            timestamp(label=f"qpu_chunk{nc}_start", timestamp_file=TIMESTAMP_FILE)
            res = stack.submit(chunk_batch)
            if on_tgcc:
                res.join()
            timestamp(label=f"qpu_chunk{nc}_finished", timestamp_file=TIMESTAMP_FILE)        
            print(f"Chunk {nc+1}/{n_chunks} done.")
            del chunk_batch
            gc.collect()

        if CONFIG["SPAM_correction"]: compute_and_save_spam_confusion_matrix(config=CONFIG)
    else:
        print("  [Build] Results already saved — extracting them from the saved files.")
    

    # ------------------------------ Merging the chunks ----------------------------------------
    
    if CONFIG["chunk_size"] is not None:
        measurement_counts = load_json(get_run_subfolder(CONFIG, "raw_results", "results_counts.json"))
        fidelity_nec_dict  = load_json(get_run_subfolder(CONFIG, "raw_results", "results_fidelity.json"))
        
        # # --- Merge chunks into flat dicts ---
        measurement_counts = merge_chunks(measurement_counts)
        fidelity_nec_dict  = merge_chunks(fidelity_nec_dict )
        
        # --- Save merged dicts for inspection ---
        merged_counts_path   = get_run_subfolder(CONFIG, "raw_results", "results_counts_merged.json")
        merged_fidelity_path = get_run_subfolder(CONFIG, "raw_results", "results_fidelity_merged.json")
        with open(merged_counts_path, "w", encoding="utf-8") as f:
            new_measurement_counts = {}
            new_measurement_counts["chunk_0"] = measurement_counts
            json.dump(new_measurement_counts, f, indent=4)
        print(f"Saved merged counts to:   {rel_path(merged_counts_path)}")
        with open(merged_fidelity_path, "w", encoding="utf-8") as f:
            new_fidelity_nec_dict = {}
            new_fidelity_nec_dict["chunk_0"] = fidelity_nec_dict
            json.dump(new_fidelity_nec_dict, f, indent=4)
        print(f"Saved merged fidelity to: {rel_path(merged_fidelity_path)}")

    else:
        # Define source paths using your existing helpers
        src_counts   = get_run_subfolder(CONFIG, "raw_results", "results_counts.json")
        src_fidelity = get_run_subfolder(CONFIG, "raw_results", "results_fidelity.json")
        dst_counts   = get_run_subfolder(CONFIG, "raw_results", "results_counts_merged.json")
        dst_fidelity = get_run_subfolder(CONFIG, "raw_results", "results_fidelity_merged.json")
    
        # Copy files
        shutil.copy2(src_counts,   dst_counts)
        shutil.copy2(src_fidelity, dst_fidelity)


    timestamp(label=f"end", timestamp_file=TIMESTAMP_FILE)
    post_processing_and_saving(CONFIG=CONFIG, roles=roles)
