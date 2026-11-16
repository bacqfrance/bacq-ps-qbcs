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


# ── NEC correction and bootstrap settings (USER_MANUAL, "NEC error mitigation and uncertainties") ─────────────
NEC_CORRECTION_MODES        = ("per_observable", "global")
NEC_CORRECTION              = "per_observable"   # the default; CONFIG["nec_correction"] overrides it
NEC_FLOOR_C                 = 2.0                # attenuation floor = NEC_FLOOR_C / sqrt(n_samples * shots_per_circuit)
NEC_BOOTSTRAP_RESAMPLES     = 1000
NEC_BOOTSTRAP_SEED          = 20261115           # one seed for every scheme, depth and SPAM mode: paired resamples
NEC_BOOTSTRAP_MIN_PER_STATE = 5                  # every NEC input state needs this many samples for a bootstrap
UNLIMITED_SHOTS             = 1e14               # the shot count the circuit plugin uses for nshots=0 (statevector)


def observable_qubits(obs_list=None, nqubits=3):
    """The measured bit positions of each observable, in the order of EXACT_OBS_LIST: each letter's position in
    "XYZ" is its qubit ("X" -> (0,), "XZ" -> (0, 2), "XYZ" -> (0, 1, 2)), bit q being character q of a
    bitstring (the convention of the counts)."""
    obs_list = EXACT_OBS_LIST if obs_list is None else obs_list
    out = []
    for name in obs_list:
        qubits = tuple("XYZ".index(c) for c in name)
        assert all(q < nqubits for q in qubits), f"observable {name} needs more than {nqubits} qubits"
        out.append(qubits)
    return out


def sign_matrix(nqubits):
    """(2**nqubits, 7) matrix: the +-1 outcome of each observable of EXACT_OBS_LIST for each bitstring
    (row int(bitstring, 2))."""
    bits = np.array([[int(c) for c in format(b, f"0{nqubits}b")] for b in range(2**nqubits)])
    return np.array([[np.prod([1 - 2 * bits[b, q] for q in S]) for S in observable_qubits(nqubits=nqubits)]
                     for b in range(2**nqubits)], dtype=float)


def _chunk_of(circuit_key):
    """The chunk number a merged circuit key ends with (merge_chunks keeps it as the index)."""
    tail = circuit_key.rsplit("_", 1)[-1]
    assert tail.isdigit(), f"Circuit key {circuit_key!r} does not end in a plain integer sample index"
    return int(tail)


def selected_chunks(job_counts, rc_samples=None):
    """The chunk numbers to analyse, in the order of the merged counts: None / an int n -> the first n; a collection
    of chunk numbers (a chunk-filtered analysis) -> the samples of those chunks."""
    chunks = [_chunk_of(k) for k in job_counts]
    if rc_samples is None or isinstance(rc_samples, (int, np.integer)):
        return chunks[:rc_samples]
    return [c for c in chunks if c in rc_samples]


def dense_counts(job_counts, nqubits, chunks):
    """One job's counts as a (len(chunks), 2**nqubits) array, row i the circuit of chunk chunks[i] (SPAM-corrected
    counts may be fractional or negative)."""
    by_chunk = {_chunk_of(k): v for k, v in job_counts.items()}
    missing = [c for c in chunks if c not in by_chunk]
    assert not missing, f"chunks {missing} are missing from a job"
    C = np.zeros((len(chunks), 2**nqubits))
    for i, c in enumerate(chunks):
        for b, n in by_chunk[c].items():
            C[i, int(b, 2)] += n
    return C


def observable_samples(C, S):
    """(n_samples, 7) expectation values of the observables per sample, from dense counts C and sign_matrix S."""
    return (C @ S) / C.sum(axis=1, keepdims=True)


def nec_attenuation_samples(C, targets, S):
    """Per NEC sample k (dense counts C, target bitstring index targets[k]):
    F (n,): the probability of the target bitstring (the global fidelity), and
    f (n, 7): the sign-corrected Z-correlator of each observable's qubits,
    f_S,k = sum_b C[k,b] prod_{q in S} (-1)^(b_q xor t_q) / total_k (the per-observable attenuation)."""
    tot = C.sum(axis=1)
    F = C[np.arange(len(C)), targets] / tot
    f = (C @ S) / tot[:, None] * S[targets]
    return F, f


def input_states(chunks, nqubits, randomized):
    """The NEC input state of each sample (chunk % 2**nqubits, as sample_index_to_input_bitstring), or 0 for all
    samples when the input state is not randomized: the bootstrap strata."""
    return np.array([c % 2**nqubits if randomized else 0 for c in chunks], dtype=int)


def samples_per_input_state(strata, nqubits, randomized):
    """The number of samples of every input state, the states with none included."""
    return [int(np.sum(strata == s)) for s in (range(2**nqubits) if randomized else [0])]


def bootstrap_weights(strata, n_resamples=NEC_BOOTSTRAP_RESAMPLES, seed=NEC_BOOTSTRAP_SEED):
    """(n_resamples, n) resample counts of a stratified bootstrap: within each stratum, as many draws with
    replacement as it has samples. A pure function of its arguments, so two analyses of the same samples
    (with and without SPAM correction) get the same resamples. The mean of a resample of X is W @ X / n."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    W = np.zeros((n_resamples, len(strata)))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        m = len(idx)
        draws = rng.integers(0, m, size=(n_resamples, m)) + m * np.arange(n_resamples)[:, None]
        W[:, idx] = np.bincount(draws.ravel(), minlength=n_resamples * m).reshape(n_resamples, m)
    return W


def attenuation_floor(n_samples, shots, c=NEC_FLOOR_C):
    """The smallest |attenuation| an NEC division uses: c / sqrt(N), N = n_samples * shots per circuit (nshots=0
    counts as the plugin's UNLIMITED_SHOTS). 1/sqrt(N) bounds the standard error of a mean of +-1 outcomes."""
    return c / np.sqrt(n_samples * (shots if shots else UNLIMITED_SHOTS))


def floor_attenuation(x, floor):
    """x with every entry |x| < floor replaced by sign(x) * floor (0 counts as positive), and the mask of the
    replaced entries."""
    x = np.asarray(x, dtype=float)
    hit = np.abs(x) < floor
    return np.where(hit, np.where(x < 0, -floor, floor), x), hit


def ci95(draws, axis=0):
    """The 95% percentile interval (2.5% and 97.5% values, stacked on a new first axis) of bootstrap draws
    (None without draws)."""
    if draws is None:
        return None
    return np.percentile(draws, [2.5, 97.5], axis=axis)


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

def compute_mitigated_expectation_values(config, rc_samples=None, roles=None, spam_samples=None, bootstrap=True):
    """
    Computes mitigated expectation values of the 7 Pauli observables for each circuit depth,
    under 6 error mitigation schemes: raw, raw+nec, rc, rc+nec, crc, crc+nec, with their bootstrap.

    The NEC schemes divide the mean of each observable over the samples by the mean of its NEC attenuation
    (a ratio of means): per observable (the default, CONFIG["nec_correction"] = "per_observable") the
    sign-corrected Z-correlator of its qubits on the NEC counts, or with "global" the probability F that the NEC
    returns its whole target bitstring (one value for all 7 observables). The attenuation is floored in absolute
    value at attenuation_floor(n_samples, shots), so a vanishing one gives a large finite value, never NaN.
    List-shaped NEC entries (runs without nec_measurement_counts) carry F only: they always use "global".

    The bootstrap resamples the RC samples within the run, stratified by NEC input state, with the same
    resamples (bootstrap_weights(), NEC_BOOTSTRAP_SEED) for every depth, role and SPAM mode. It runs only if
    bootstrap is True and every input state has at least NEC_BOOTSTRAP_MIN_PER_STATE samples.

    Args:
        config (dict):      CONFIG dict with T_list, dt, folder_path, etc.
        rc_samples (int):   Number of RC circuits to average over (None = all); may also be a set of
                             chunk numbers (a chunk-filtered analysis).
        spam_samples (int): Number of SPAM calibration chunks for the SPAM correction.
                             None (default) = the maximum, every collected chunk:
                             the run's own spam_confusion_matrix_inv.npy, whatever
                             an integer rc_samples is; when rc_samples is a chunk set, None
                             means the matrix of those kept chunks. Only a value below the number of
                             collected chunks loads (building it if missing) the
                             subset matrix spam_confusion_matrix_inv+nsamples-{spam_samples}.npy.
        bootstrap (bool):   False skips the bootstrap.

    Returns:
        (results_per_depth, nec_info):
        results_per_depth: dict keyed by trotter steps (int), each containing
            { "raw", "raw_nec", "rc", "rc_nec", "crc", "crc_nec" }:
                { "mean": np.ndarray (7,), "boot": np.ndarray (n_resamples, 7) or None }
            and for the NEC roles also "attenuation" (7,) (before the floor), "floor_hits" (how many of the 7
            full-run attenuations were floored) and "floor_hit_rate_bootstrap" (the fraction of floored
            attenuations over all resamples, None without a bootstrap);
        nec_info: the settings and bookkeeping that run_data*.json records as "nec_analysis".
    """
    measurement_counts = load_json(get_run_subfolder(config, "raw_results", "results_counts_merged.json"))["chunk_0"]
    fidelity_nec_dict  = load_json(get_run_subfolder(config, "raw_results", "results_fidelity_merged.json"))["chunk_0"]

    if config.get("SPAM_correction", False):
        if spam_samples is None and not isinstance(rc_samples, (int, type(None))):
            spam_samples = rc_samples   # a chunk-filtered analysis: the SPAM matrix of the kept chunks only
        by_chunks = spam_samples is not None and not isinstance(spam_samples, int)
        subset   = by_chunks or (spam_samples is not None and spam_samples < count_spam_calibration_chunks(config))
        suffix   = subset_suffix(spam_samples) if subset else ""
        M_inv_path = get_run_subfolder(config, "processed_results", f"spam_confusion_matrix_inv{suffix}.npy")
        if by_chunks or not os.path.exists(M_inv_path):   # the matrix of a chunk set is rebuilt every time
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

    mode = config.get("nec_correction") or NEC_CORRECTION
    assert mode in NEC_CORRECTION_MODES, f"nec_correction must be one of {NEC_CORRECTION_MODES}, got {mode!r}"
    nq         = config["nsites"]
    S          = sign_matrix(nq)
    randomized = bool(config["nec_randomize_input_state"])
    n_nec_input_states = config["samples"] if randomized else 1
    shots      = config.get("shots", 0)

    chunks = W = None
    mode_used = mode
    results_per_depth = {}

    for di, nt in enumerate(trotter_steps_list):
        tag        = f"Nt_{nt:02d}"

        classical_nec_map = load_classical_nec_map(config, tag) if config["nec_measurement_counts"] else None

        role_keys = {}
        for ri, role in enumerate(roles):
            local_idx = di * n_roles + ri
            role_keys[role] = f"job_{tag}_{role}_ems{local_idx}"

        # The analysed samples: the same chunks at every depth, so one set of resamples serves all of them
        first_job = measurement_counts[role_keys[non_nec_roles[0]]]
        depth_chunks = selected_chunks(first_job, rc_samples)
        if chunks is None:
            chunks     = depth_chunks
            all_chunks = [_chunk_of(k) for k in first_job]
            n          = len(chunks)
            strata     = input_states(chunks, nq, randomized)
            per_state  = samples_per_input_state(strata, nq, randomized)
            do_boot    = bootstrap and min(per_state) >= NEC_BOOTSTRAP_MIN_PER_STATE
            W          = bootstrap_weights(strata) if do_boot else None
            floor      = attenuation_floor(n, shots)
        assert depth_chunks == chunks, f"{tag}: the analysed chunks differ from the first depth's"

        results_per_depth[nt] = {}
        for role, nec_role in zip(non_nec_roles, nec_roles):
            obs      = observable_samples(dense_counts(measurement_counts[role_keys[role]], nq, chunks), S)
            obs_mean = obs.mean(axis=0)
            obs_boot = W @ obs / n if W is not None else None
            results_per_depth[nt][role] = {"mean": obs_mean, "boot": obs_boot}

            entry = fidelity_nec_dict[role_keys[nec_role]]
            if config["nec_measurement_counts"]:
                assert isinstance(entry, dict), f"{role_keys[nec_role]}: expected NEC counts, got a list"
            if isinstance(entry, dict):
                targets = np.array([int(get_target_bitstring(f"c_{c}", n_nec_input_states, nq, classical_nec_map), 2)
                                    for c in chunks])
                F, f = nec_attenuation_samples(dense_counts(entry, nq, chunks), targets, S)
                att = f if mode == "per_observable" else np.repeat(F[:, None], S.shape[1], axis=1)
            else:   # one fidelity per sample, in the order of the counts: no per-observable attenuation
                pos = {c: i for i, c in enumerate(all_chunks)}
                F = np.array([entry[pos[c]] for c in chunks if pos[c] < len(entry)], dtype=complex).real
                att = np.repeat(F[:, None], S.shape[1], axis=1)
                mode_used = "global"
            att_mean = att.mean(axis=0)
            att_floored, hit = floor_attenuation(att_mean, floor)
            out = {"mean": obs_mean / att_floored, "boot": None, "attenuation": att_mean,
                   "floor_hits": int(hit.sum()), "floor_hit_rate_bootstrap": None}
            if W is not None:
                att_boot, hit_boot = floor_attenuation(W @ att / n, floor)
                out["boot"] = obs_boot / att_boot
                out["floor_hit_rate_bootstrap"] = float(hit_boot.mean())
            results_per_depth[nt][nec_role] = out

    nec_info = {
        "nec_correction":                mode_used,
        "attenuation_floor_c":           NEC_FLOOR_C,
        "attenuation_floor":             float(floor),
        "n_samples":                     n,
        "shots_per_circuit":             shots,
        "bootstrap_done":                W is not None,
        "bootstrap_resamples":           NEC_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed":                NEC_BOOTSTRAP_SEED,
        "bootstrap_stratified":          True,
        "bootstrap_min_per_input_state": NEC_BOOTSTRAP_MIN_PER_STATE,
        "samples_per_input_state_min":   min(per_state),
        "interval":                      "percentile: the 2.5% and 97.5% values of the bootstrap",
    }
    if W is None:
        nec_info["bootstrap_skipped_because"] = (
            "bootstrap switched off" if not bootstrap else
            f"no interval: n = {n}, fewer than {NEC_BOOTSTRAP_MIN_PER_STATE} samples per input state")
    return results_per_depth, nec_info


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
        ideal_values : array of shape (7,) - noiseless expectation values
        noisy_values : array of shape (..., 7) - noisy expectation values (leading axes, e.g. bootstrap draws,
                       give one value each)

    Returns:
        epsilon : float, or an array of the leading shape - figure of merit
    """
    ideal_values = np.asarray(ideal_values, dtype=float)
    noisy_values = np.asarray(noisy_values, dtype=float)
    assert ideal_values.shape == (7,) and noisy_values.shape[-1] == 7

    S = np.sum(np.abs(ideal_values))
    return np.sum(np.abs(ideal_values) * np.abs(noisy_values - ideal_values), axis=-1) / S


def post_processing_and_saving(CONFIG, roles=None, use_circuit_LNN=False, filename="run_data.json", n_samples=None, print_runtimes=True, qiskit_v13_qpy=False,
                             spam_n_samples=None):

    '''Post processing to save some results after the runs.

    n_samples: analyze only the first n_samples chunks instead of the full
    CONFIG["samples"], or only the chunks of a chunk_scope.HealthCheckScope (the ones whose extra
    health checks passed; the file names then carry "+hc-gt-{threshold}"). None (default) reproduces the original full-run
    behaviour and filenames exactly. A genuine subset (co-existing with the
    full run's own processed_results, never overwriting them) suffixes the
    output filename with "+nsamples-{n_samples}".

    spam_n_samples: SPAM calibration chunks for the SPAM correction. None
    (default) = the maximum, every collected chunk (the run's own SPAM
    confusion matrix), also for a subset analysis: the subset varies only the
    RC samples; for a chunk-filtered analysis (n_samples a HealthCheckScope) it is
    the matrix of the kept chunks. Only a value below the collected chunks builds a
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
        if spam_n_samples is None and not isinstance(effective_n, int):
            spam_n_samples = effective_n   # a chunk-filtered analysis: the SPAM matrix of the kept chunks only
        if spam_n_samples is not None and not isinstance(spam_n_samples, int):
            spam_used = len(spam_n_samples)
            compute_and_save_spam_confusion_matrix(CONFIG, n_samples=spam_n_samples)
        else:
            spam_used = spam_n_samples if spam_n_samples is not None and spam_n_samples < n_spam_available else n_spam_available
            if spam_used < n_spam_available:
                compute_and_save_spam_confusion_matrix(CONFIG, n_samples=spam_used)

    mitigated_results, nec_info = compute_mitigated_expectation_values(CONFIG, rc_samples=effective_n, roles=roles,
                                                                       spam_samples=spam_n_samples)
    nchunks = effective_n if isinstance(effective_n, int) else len(effective_n)   # a chunk-filtered scope is a set

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
    run_data["n_samples_analyzed"] = nchunks
    if spam_used is not None:
        run_data["spam_calibration_samples"] = spam_used
    # What was analyzed and what the run still misses (uncollected chunks / health checks)
    run_data["completeness"]       = completeness(run_path, CONFIG["samples"], effective_n if subset else None)
    if completeness_warning(run_data["completeness"]) and print_runtimes:
        print(f"  {completeness_warning(run_data['completeness'])}")
    run_data["nec_analysis"]       = nec_info
    run_data["depths"]             = {}
    obs_list     = list(EXACT_OBS_LIST)
    nts          = [int(nt) for nt in mitigated_results]
    exact_values = load_exact_observables(CONFIG, nts, use_circuit_LNN)
    Z_exact      = np.array([float(np.sum(np.abs(exact_values[nt]))) for nt in nts])
    shots_per_sample = int(round(CONFIG["shots"]))

    # Per scheme: the full-run AWAE per depth and, with a bootstrap, its (n_resamples, n_depths) draws; the
    # cumulative AWAE of every draw uses the same resamples at all depths (paired).
    from source.core.analysis.derived_quantities import cumulative_awae, fe_quantities
    schemes = [(ems, role) for ems, role in zip(BASE_EMS_LIST, ["raw", "raw_nec", "rc", "rc_nec", "crc", "crc_nec"])
               if role in roles]
    awae, awae_boot, cumul, cumul_boot = {}, {}, {}, {}
    for ems, role in schemes:
        awae[ems]  = np.array([compute_AWAE(exact_values[nt], mitigated_results[nt][role]["mean"]) for nt in nts])
        cumul[ems] = cumulative_awae(Z_exact, awae[ems])
        if nec_info["bootstrap_done"]:
            awae_boot[ems]  = np.stack([compute_AWAE(exact_values[nt], mitigated_results[nt][role]["boot"])
                                        for nt in nts], axis=-1)
            cumul_boot[ems] = cumulative_awae(Z_exact, awae_boot[ems])

    def _ci(draws, i):   # [lo, hi] at depth index i, None without a bootstrap
        return None if draws is None else [float(v) for v in ci95(draws[:, i])]

    def _median(draws, i):
        return None if draws is None else float(np.median(draws[:, i]))

    # E_bare, FE_coh_nec and FE_ct_nec per bootstrap draw (their intervals; the summary picks the depth)
    fe_boot = None
    if nec_info["bootstrap_done"] and all(e in cumul_boot for e in ("bare", "NEC", "NEC+RC", "NEC+cRC")):
        fe_boot = fe_quantities(cumul_boot["bare"], cumul_boot["NEC"], cumul_boot["NEC+RC"], cumul_boot["NEC+cRC"])

    for i, nt in enumerate(nts):
        depth = run_data["depths"][nt] = {
            "Z_exact"                   : float(Z_exact[i]),
            "perfect_circuit_obs_values": np.real(exact_values[nt]).tolist(),
        }
        for ems, role in schemes:
            res = mitigated_results[nt][role]
            entry = depth[ems] = {
                "obs_values":        np.real(res["mean"]).tolist(),
                "AWAE":              float(awae[ems][i]),            # full run (the point estimate)
                "AWAE_median":       _median(awae_boot.get(ems), i),  # bootstrap median
                "AWAE_ci95":         _ci(awae_boot.get(ems), i),      # bootstrap 95% percentile interval
                "AWAE_cumul":        float(cumul[ems][i]),
                "AWAE_cumul_median": _median(cumul_boot.get(ems), i),
                "AWAE_cumul_ci95":   _ci(cumul_boot.get(ems), i),
                "shots_per_sample":  shots_per_sample,
            }
            if "attenuation" in res:
                entry["nec_attenuation"]                   = np.real(res["attenuation"]).tolist()
                entry["attenuation_floor_hits"]            = res["floor_hits"]
                entry["attenuation_floor_hit_rate_bootstrap"] = res["floor_hit_rate_bootstrap"]
        depth["derived_ci95"] = {k: _ci(v, i) for k, v in fe_boot.items()} if fe_boot is not None else None


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
