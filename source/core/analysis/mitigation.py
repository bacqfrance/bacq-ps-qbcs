"""
mitigation.py
From measured counts to results: expectation values, the NEC / RC / cRC mitigation, merging the
chunks, the AWAE figure of merit, and post_processing_and_saving(), which writes a run's
processed_results/run_data*.json.
"""

import json
import os
from pathlib import Path

import numpy as np

from source.core.analysis.run_timing import aggregate_run_timing
from source.core.circuits.SPAM_functions import (apply_spam_correction_nested, compute_and_save_spam_confusion_matrix,
                                                 count_spam_calibration_chunks)
from source.core.circuits.building import EXACT_OBS_LIST, load_exact_observables
from source.core.common import get_circuit_folder, get_run_subfolder, get_target_bitstring, load_json
from source.core.records.chunk_scope import completeness, completeness_warning
from source.core.records.job_records import saved_config
from source.core.records.run_names import subset_suffix

# The base mitigation schemes of a run_data*.json depth entry (a SPAM-corrected file has the same names)
BASE_EMS_LIST = ["bare", "NEC", "RC", "NEC+RC", "cRC", "NEC+cRC"]


def calc_expval_measurement_dict(measurement_counts):
    """
    Computes expectation values of 7 Pauli observables (X0, Y1, Z2, X0Y1, X0Z2, Y1Z2, X0Y1Z2)
    from measurement counts over multiple jobs and randomized compilation (RC) circuits.

    Args:
        measurement_counts (dict): Nested dict { job_key: { rc_key: { bitstring: count } } }
            - bitstring: 3-character binary string (e.g. "010")

    Returns:
        - avg_value_array (np.ndarray): Shape (n_jobs, n_rc_circuits, 7) 
        - per-circuit expectation values (np.ndarray): Shape (n_jobs, 7) — expectation values averaged over RC circuits per job.
    """
    n_jobs = len(measurement_counts)
    circuit_counts = {job_key: len(rc_dict) for job_key, rc_dict in measurement_counts.items()}
    n_rc_circuits = next(iter(circuit_counts.values()))
    assert all(c == n_rc_circuits for c in circuit_counts.values()), \
        f"Jobs have inconsistent RC-circuit counts, expected uniform {n_rc_circuits}: {circuit_counts}"

    # Shape: (n_jobs, n_rc_circuits, 7)
    avg_value_array = np.zeros((n_jobs, n_rc_circuits, 7), dtype=complex)

    for ij, job_key in enumerate(measurement_counts):
        results_rc_circuits = measurement_counts[job_key]

        for ir, rc_key in enumerate(results_rc_circuits):
            counts = results_rc_circuits[rc_key]
            total = sum(counts.values())

            X0 = Y1 = Z2 = X0Y1 = X0Z2 = Y1Z2 = X0Y1Z2 = 0

            for ky, it in counts.items():
                s0 = 1 - 2 * int(ky[0])
                s1 = 1 - 2 * int(ky[1])
                s2 = 1 - 2 * int(ky[2])

                X0     += it * s0
                Y1     += it * s1
                Z2     += it * s2
                X0Y1   += it * s0 * s1
                X0Z2   += it * s0 * s2
                Y1Z2   += it * s1 * s2
                X0Y1Z2 += it * s0 * s1 * s2

            avg_value_array[ij, ir, :] = np.array(
                [X0, Y1, Z2, X0Y1, X0Z2, Y1Z2, X0Y1Z2]
            ) / total

    return avg_value_array, avg_value_array.mean(axis=1)


class analyze_measurement_outcomes:

    def __init__(self, dict_measurement_counts, nqubits=None):
        self.dict_measurement_counts = dict_measurement_counts
        self.measurement_data,_ = calc_expval_measurement_dict(dict_measurement_counts)
        self.njobs = self.measurement_data.shape[0]
        self.rc_circuits_per_job = self.measurement_data.shape[1]
        assert nqubits is not None,"nqubits cannot be None!!"
        self.nqubits = nqubits

    def compute_expectation_values_per_job(self, job_index=0, rc_samples=None):
        if rc_samples is None: 
            rc_samples = self.rc_circuits_per_job # Full average
        
        mean_ = self.measurement_data[job_index, :rc_samples, :].mean(axis=0)       # (7)
        std_  = self.measurement_data[job_index, :rc_samples, :].std(axis=0)        # (7)

        return mean_, std_

    def compute_average_fidelity_nec(self, job_id, fidelity_nec_dict, 
                                     rc_samples=None, classical_nec_map=None, 
                                     expect_counts=None, n_input_states=None):

        entry = fidelity_nec_dict[job_id]
        is_dict_shaped = isinstance(entry, dict)

        if expect_counts is not None:
            assert is_dict_shaped == expect_counts, \
                f"{job_id}: got {'dict' if is_dict_shaped else 'list'}-shaped entry, but expect_counts={expect_counts}"

        if rc_samples is None:
            rc_samples = len(entry) if is_dict_shaped else self.rc_circuits_per_job  # Full average

        if is_dict_shaped:
            assert classical_nec_map is not None and n_input_states is not None, \
                "classical_nec_map and n_input_states must be provided when fidelity_nec_dict entries are counts-based"
            rc_items = list(entry.items())[0:rc_samples]
            vals = []
            for rc_key, counts in rc_items:
                total = sum(counts.values())
                target = get_target_bitstring(rc_key, n_input_states, self.nqubits, classical_nec_map)
                vals.append(counts.get(target, 0) / total)
            vals = np.array(vals, dtype=float)

        else:
            vals = np.array(entry[0:rc_samples], dtype=complex)

        return vals.mean(), vals.std()

def load_classical_nec_map(config, depth_label):
    path = os.path.join(get_circuit_folder(config, top_folder=True), f"nec_classical_map_{depth_label}.json")
    raw = load_json(path)
    assert len(raw) == 1, \
        f"Expected exactly one job's classical map per depth, got {len(raw)} entries: {list(raw.keys())}"
    return next(iter(raw.values()))

def merge_chunks(chunked_dict):
    """
    Merges chunk_0, chunk_1, ... into a flat dict, concatenating lists with same keys.
    Works on both fidelity and counts dictionaries.
    If no chunk keys found, assumes already flat (no chunking) and returns as-is.
    """
    chunk_keys = sorted(
    [k for k in chunked_dict.keys() if k.startswith("chunk_")],
    key=lambda x: int(x.split("_")[1])  # sort by number, not string
    )
    if not chunk_keys:
        return chunked_dict  # already flat, no chunking was used

    # One circuit per job key and chunk (the chunk is the sample): the merged circuit keeps
    # its chunk number as index, so a run with uncollected chunks in the middle keeps each
    # sample's own index — the NEC analysis derives the sample's input state from it.
    by_chunk = all(len(v) == 1 for ck in chunk_keys for v in chunked_dict[ck].values() if isinstance(v, dict))
    merged = {}
    circuit_counters = {}
    for chunk_key in chunk_keys:
        for k, v in chunked_dict[chunk_key].items(): # job
            if k not in merged:
                merged[k] = {} if isinstance(v, dict) else []
                circuit_counters[k] = 0

            if isinstance(v, dict):
                # re-index circuit keys to be globally unique
                for circuit_key, circuit_val in v.items(): # circuit instance
                    # extract the prefix e.g. "rc_circuit" from "rc_circuit_0"
                    prefix = "_".join(circuit_key.split("_")[:-1])
                    index   = int(chunk_key.split("_")[1]) if by_chunk else circuit_counters[k]
                    new_key = f"{prefix}_{index}"
                    merged[k][new_key] = circuit_val
                    circuit_counters[k] += 1
            else:
                merged[k].extend(v)

    return merged

def compute_mitigated_expectation_values(config, rc_samples=None, roles=None, spam_samples=None):
    """
    Computes mitigated expectation values of 7 Pauli observables for each circuit depth,
    under 6 error mitigation schemes: raw, raw+nec, rc, rc+nec, crc, crc+nec.

    Args:
        config (dict):      CONFIG dict with T_list, dt, folder_path, etc.
        rc_samples (int):   Number of RC circuits to average over (None = all).
        spam_samples (int): Number of SPAM calibration chunks for the SPAM correction.
                             None (default) = the maximum, every collected chunk:
                             the run's own spam_confusion_matrix_inv.npy, whatever
                             rc_samples is. Only a value below the number of
                             collected chunks loads (building it if missing) the
                             subset matrix spam_confusion_matrix_inv+nsamples-{spam_samples}.npy.

    Returns:
        dict keyed by trotter steps (int), each containing:
            { "raw", "raw_nec", "rc", "rc_nec", "crc", "crc_nec" }:
                { "mean": np.ndarray (7,), "std": np.ndarray (7,) }
    """
    measurement_counts = load_json(get_run_subfolder(config, "raw_results", "results_counts_merged.json"))["chunk_0"]
    fidelity_nec_dict  = load_json(get_run_subfolder(config, "raw_results", "results_fidelity_merged.json"))["chunk_0"]

    if config.get("SPAM_correction", False):
        subset   = spam_samples is not None and spam_samples < count_spam_calibration_chunks(config)
        suffix   = subset_suffix(spam_samples) if subset else ""
        M_inv_path = get_run_subfolder(config, "processed_results", f"spam_confusion_matrix_inv{suffix}.npy")
        if not os.path.exists(M_inv_path):
            # Full-run matrix is normally built once during collect_results();
            # regenerate it here if e.g. processed_results/ was cleared, since
            # the calibration counts it's built from live in raw_results/.
            compute_and_save_spam_confusion_matrix(config, n_samples=spam_samples if subset else None)
        M_inv  = np.load(M_inv_path)
        measurement_counts = apply_spam_correction_nested(measurement_counts, M_inv, config["nsites"])
        fidelity_nec_dict  = apply_spam_correction_nested(fidelity_nec_dict,  M_inv, config["nsites"])

    trotter_steps_list = np.round(np.array(config["T_list"]) / config["dt"]).astype(int)
    if roles is None:
        roles = ["raw", "raw_nec", "rc", "rc_nec", "crc", "crc_nec"]
    n_roles = len(roles)

    non_nec_roles = [role for role in roles if not role.endswith("_nec")]
    nec_roles     = [role for role in roles if role.endswith("_nec")]
    assert all(role == nec_role[:-4] for role, nec_role in zip(non_nec_roles, nec_roles)),(
        f"Role pairing mismatch: {list(zip(non_nec_roles, nec_roles))}"
    )

    results_per_depth = {}

    for di, nt in enumerate(trotter_steps_list):
        tag        = f"Nt_{nt:02d}"

        classical_nec_map = load_classical_nec_map(config, tag) if config["nec_measurement_counts"] else None

        role_keys = {}
        for ri, role in enumerate(roles):
            local_idx = di * n_roles + ri
            role_keys[role] = f"job_{tag}_{role}_ems{local_idx}"

        # Build per-depth counts dict for analyzer
        depth_counts = {
            role_keys[role]: measurement_counts[role_keys[role]]
            for role in non_nec_roles
            if role_keys[role] in measurement_counts
        }
        analyzer = analyze_measurement_outcomes(depth_counts, nqubits=config["nsites"])

        # ── Expectation values per non-NEC role ───────────────────────────────
        exp_values = {}
        for ji, role in enumerate(non_nec_roles):
            mean, std          = analyzer.compute_expectation_values_per_job(
                                    job_index=ji, rc_samples=rc_samples)
            exp_values[role]   = {"mean": mean, "std": std}

        n_nec_input_states = config["samples"] if config["nec_randomize_input_state"] else 1

        # ── NEC fidelity per NEC role ─────────────────────────────────────────
        fid_values = {}
        for role in nec_roles:
            fid_mean, fid_std  = analyzer.compute_average_fidelity_nec(
                job_id            = role_keys[role],
                fidelity_nec_dict = fidelity_nec_dict,
                rc_samples        = rc_samples,
                classical_nec_map = classical_nec_map,
                n_input_states    = n_nec_input_states,
                expect_counts     = config["nec_measurement_counts"]
            )
            fid_values[role]   = {"mean": fid_mean, "std": fid_std}

        # ── Assemble results for this depth ───────────────────────────────────
        results_per_depth[nt] = {}
        for role, nec_role in zip(non_nec_roles, nec_roles):
            assert role==nec_role[:-4], f"{role} and {nec_role} mismatch!!"
            results_per_depth[nt][role]     = {
                "mean": exp_values[role]["mean"].real,
                "std":  exp_values[role]["std"].real
            }
            results_per_depth[nt][nec_role] = {
                "mean": (exp_values[role]["mean"] / fid_values[nec_role]["mean"]).real,
                "std":   fid_values[nec_role]["std"].real
            }

    return results_per_depth


def check_shots_consistency(config):
    """
    Spot-checks one job/circuit in raw_results/results_counts_merged.json
    against config["shots"] and prints a warning on mismatch — a guard
    against accidentally analyzing a run_folder that was actually submitted
    with a different nshots than the CONFIG currently in hand.

    Non-fatal: only warns, never raises, and never blocks analysis.
    """
    if not config.get("shots", 0):
        return   # nshots=0 (statevector) has no shot count to check

    try:
        counts   = load_json(get_run_subfolder(config, "raw_results", "results_counts_merged.json"))["chunk_0"]
        job_key  = next(iter(counts))
        circ_key = next(iter(counts[job_key]))
        observed_shots = sum(counts[job_key][circ_key].values())
    except (FileNotFoundError, KeyError, StopIteration):
        print("  ⚠️  Could not verify shot count from raw_results — skipping check.")
        return

    if observed_shots != config["shots"]:
        print(f"  ⚠️  Shot-count mismatch: raw_results shows {observed_shots} shots for "
              f"'{job_key}'/'{circ_key}', but this instance is configured with "
              f"nshots={config['shots']}. Are you analyzing the right run_folder?")


def compute_AWAE(ideal_values, noisy_values):
    """
    Computes epsilon = (1/S) * sum_j |i_j| * |n_j - i_j|
    where S = sum_j |i_j|
    
    Args:
        ideal_values : array of shape (n_obs,) - noiseless expectation values
        noisy_values : array of shape (n_obs,) - noisy expectation values
    
    Returns:
        epsilon : float - figure of merit
    """
    assert len(ideal_values)==len(noisy_values)
    assert len(ideal_values)==7

    ideal_values = np.array(ideal_values)
    noisy_values = np.array(noisy_values)

    S       = np.sum(np.abs(ideal_values))
    epsilon = (1 / S) * np.sum(np.abs(ideal_values) * np.abs(noisy_values - ideal_values))

    return epsilon


def post_processing_and_saving(CONFIG, roles=None, use_circuit_LNN=False, filename="run_data.json", n_samples=None, print_runtimes=True, qiskit_v13_qpy=False,
                             spam_n_samples=None):

    '''Post processing to save some results after the runs.

    n_samples: analyze only the first n_samples chunks instead of the full
    CONFIG["samples"]. None (default) reproduces the original full-run
    behaviour and filenames exactly. A genuine subset (co-existing with the
    full run's own processed_results, never overwriting them) suffixes the
    output filename with "+nsamples-{n_samples}".

    spam_n_samples: SPAM calibration chunks for the SPAM correction. None
    (default) = the maximum, every collected chunk (the run's own SPAM
    confusion matrix), also for a subset analysis: the subset varies only the
    RC samples. Only a value below the collected chunks builds a
    "+nsamples-{spam_n_samples}" SPAM matrix. run_data_spam*.json records the
    chunks used as "spam_calibration_samples".

    qiskit_v13_qpy: pass through to aggregate_run_timing() — read
    the circuit-build time from the QPY-v13 timestamps log instead of the
    default one, for runs on all_to_all circuits (see bacq_ps_qbcs.build()).
    '''

    trotter_steps_list = np.round(np.array(CONFIG["T_list"]) / CONFIG["dt"]).astype(int)
    if roles is None:
        roles = ["raw", "raw_nec", "rc", "rc_nec", "crc", "crc_nec"]
    assert len(roles) * len(trotter_steps_list) == CONFIG["chunk_size"]
    assert len(CONFIG["epsilon"])==CONFIG["nsites"]
    assert len(CONFIG["local_qubit_basis"])==CONFIG["nsites"]

    effective_n = n_samples if n_samples is not None else CONFIG["samples"]
    subset      = effective_n != CONFIG["samples"]

    spam_used = None
    if CONFIG.get("SPAM_correction", False):
        n_spam_available = count_spam_calibration_chunks(CONFIG)
        spam_used = spam_n_samples if spam_n_samples is not None and spam_n_samples < n_spam_available else n_spam_available
        if spam_used < n_spam_available:
            compute_and_save_spam_confusion_matrix(CONFIG, n_samples=spam_used)

    mitigated_results          = compute_mitigated_expectation_values(CONFIG, rc_samples=effective_n, roles=roles,
                                                                      spam_samples=spam_n_samples)
    nchunks = effective_n

    # CPU / QPU / queue time (health checks included) — the same aggregate_run_timing()
    # summary.json uses, so both report and print the same numbers — and the
    # executed-vs-submitted circuit checks, chunks and health checks separately
    # (source/core/records/circuit_check.py).
    from source.core.records.circuit_check import summarize_circuit_checks, print_verification
    run_data = {}
    run_path = Path(CONFIG["folder_path"])
    try:
        timing = aggregate_run_timing(str(run_path.resolve()), n_samples=effective_n if subset else None,
                                      total_chunks=CONFIG["samples"], qiskit_v13_qpy=qiskit_v13_qpy, verbose=print_runtimes)
    except FileNotFoundError as e:           # no run_info.json (very old runs)
        print(f"  ⚠️  timing breakdown skipped: {e}")
        timing = None
    # The short keys of older run_data files, now taken from the same breakdown:
    # build (myQLM + Qiskit conversion), the rest of the CPU time (transpilation + submission,
    # health checks included), QPU time and queue time.
    cpu = (timing or {}).get("cpu") or {}
    run_data["build_time_s"]            = cpu.get("build_total_s")
    run_data["transpile_submit_time_s"] = (cpu["total_s"] - (cpu.get("build_total_s") or 0.0)) if cpu else None
    run_data["qpu_runtime_s"]           = timing["failed_attempts"]["totals_incl_failed"]["qpu_s"] if timing else None   # failed jobs included
    run_data["queue_time_s"]            = timing["failed_attempts"]["totals_incl_failed"]["queue_s"] if timing else None   # failed jobs included
    if timing is not None:
        run_data["timing"] = timing
    run_data["circuit_verification"] = summarize_circuit_checks(run_path, n_samples=effective_n if subset else None)
    print_verification(run_data["circuit_verification"])
    run_data["nchunks"]            = nchunks
    run_data["n_samples_analyzed"] = effective_n
    if spam_used is not None:
        run_data["spam_calibration_samples"] = spam_used
    # What was analyzed and what the run still misses (uncollected chunks / health checks)
    run_data["completeness"]       = completeness(run_path, CONFIG["samples"], effective_n if subset else None)
    if completeness_warning(run_data["completeness"]) and print_runtimes:
        print(f"  {completeness_warning(run_data['completeness'])}")
    run_data["depths"]             = {}
    obs_list     = list(EXACT_OBS_LIST)
    exact_values = load_exact_observables(CONFIG, list(mitigated_results), use_circuit_LNN)
    for nt, results in mitigated_results.items():
        # ── Exact observables for this specific depth ─────────────────────
        T_nt = nt * CONFIG["dt"]
        exact_obs_values_nt = exact_values[int(nt)]
        Z_exact_nt = float(np.sum(np.abs(exact_obs_values_nt)))

        obs_values = {
            role: np.real(results[role]["mean"])
            for role in results.keys()
        }

        bare_obs_values       = obs_values.get("raw",     None)
        nec_mitigated_obs     = obs_values.get("raw_nec", None)
        rc_mitigated_obs      = obs_values.get("rc",      None)
        rc_nec_mitigated_obs  = obs_values.get("rc_nec",  None)
        crc_mitigated_obs     = obs_values.get("crc",     None)
        crc_nec_mitigated_obs = obs_values.get("crc_nec", None)

 
        shots_per_sample = int(round(CONFIG["shots"]))

        run_data["depths"][int(nt)] = {
            "Z_exact"                   : Z_exact_nt,
            "perfect_circuit_obs_values": np.real(exact_obs_values_nt).tolist(),
        }
        if bare_obs_values       is not None: run_data["depths"][int(nt)]["bare"]    = {"obs_values": bare_obs_values.tolist(),       "AWAE": compute_AWAE(exact_obs_values_nt, bare_obs_values).tolist(),       "shots_per_sample": shots_per_sample}
        if nec_mitigated_obs     is not None: run_data["depths"][int(nt)]["NEC"]     = {"obs_values": nec_mitigated_obs.tolist(),     "AWAE": compute_AWAE(exact_obs_values_nt, nec_mitigated_obs).tolist(),     "shots_per_sample": shots_per_sample}
        if rc_mitigated_obs      is not None: run_data["depths"][int(nt)]["RC"]      = {"obs_values": rc_mitigated_obs.tolist(),      "AWAE": compute_AWAE(exact_obs_values_nt, rc_mitigated_obs).tolist(),      "shots_per_sample": shots_per_sample}
        if rc_nec_mitigated_obs  is not None: run_data["depths"][int(nt)]["NEC+RC"]  = {"obs_values": rc_nec_mitigated_obs.tolist(),  "AWAE": compute_AWAE(exact_obs_values_nt, rc_nec_mitigated_obs).tolist(),  "shots_per_sample": shots_per_sample}
        if crc_mitigated_obs     is not None: run_data["depths"][int(nt)]["cRC"]     = {"obs_values": crc_mitigated_obs.tolist(),     "AWAE": compute_AWAE(exact_obs_values_nt, crc_mitigated_obs).tolist(),     "shots_per_sample": shots_per_sample}
        if crc_nec_mitigated_obs is not None: run_data["depths"][int(nt)]["NEC+cRC"] = {"obs_values": crc_nec_mitigated_obs.tolist(), "AWAE": compute_AWAE(exact_obs_values_nt, crc_nec_mitigated_obs).tolist(), "shots_per_sample": shots_per_sample}

 
    CONFIG["epsilon"]               = CONFIG["epsilon"].tolist()
    CONFIG["noise_angles"]          = CONFIG["noise_angles"].tolist()
    CONFIG["neighbor_noise_angles"] = CONFIG["neighbor_noise_angles"].tolist()
    run_data["config"]   = saved_config(CONFIG)   # no machine-specific absolute path in the data
    run_data["obs_list"] = obs_list
    out_filename = filename.replace(".json", f"{subset_suffix(effective_n)}.json") if subset else filename
    with open(get_run_subfolder(CONFIG, "processed_results", out_filename), "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4)
    # reconverting the mutated objects
    CONFIG["epsilon"]               = np.array(CONFIG["epsilon"])
    CONFIG["noise_angles"]          = np.array(CONFIG["noise_angles"])
    CONFIG["neighbor_noise_angles"] = np.array(CONFIG["neighbor_noise_angles"])
