# myQLM is imported only inside build_spam_calibration_batch(), so the SPAM
# correction code loads without it (real-QPU venvs, analysis).
import numpy as np
from source.core.common import get_run_subfolder
from source.core.records.run_names import subset_suffix
import json 
import os 
import warnings

def build_spam_calibration_batch(nqubits, shots_per_state):
    """
    Builds a Batch of 2**nqubits calibration jobs, one per computational
    basis state (in order 0...2**nqubits-1).
    """
    from qat.core import Batch
    from qat.lang.AQASM import Program
    from source.core.circuits.main import prep_computational_basis_input
    def build_spam_calibration_job(nqubits, input_bitstring, shots):
        """
        A single calibration job for a given computational-basis state.
        """
        prog = Program()
        q = prog.qalloc(nqubits)
        prep_computational_basis_input(prog, q, input_bitstring)
        job = prog.to_circ().to_job(nbshots=shots)
        job.meta_data = {"role": "spam_cal", "input_state": input_bitstring}
        return job

    jobs = [
        build_spam_calibration_job(nqubits, format(state, f"0{nqubits}b"), shots_per_state)
        for state in range(2**nqubits)
    ]
    return Batch(jobs=jobs)

# ================================================ AFTER SPAM submission ======================================================================= #
def extract_spam_calibration_counts(batch_result, nqubits, shots_per_state, chunk_id=None, config=None, append_counts_dict=True):
    """
    Turns a BatchResult from a SPAM calibration batch (one job per prepared
    basis state, in ascending order) into
        { "chunk_<chunk_id>": { input_bitstring: { measured_bitstring: count } } }
    saves it in raw_results/spam_calibration_counts_per_chunk.json (added to the
    existing file when append_counts_dict) and returns it.
    """
    assert chunk_id is not None, "There has to be a chunk id!!"
    assert config is not None, "There has to be a config dict!!"
    assert len(batch_result.results)==2**nqubits, "The length of the results object must be equal to the 2**nqubits!!"

    counts_per_input = {}
    for ind, result in enumerate(batch_result.results):
        input_bitstring = format(ind, f"0{nqubits}b") # results are in order of the basis states by construction
        counts = {}
        for s in result:
            meas_state = int.from_bytes(s._state, "big")
            bitstring  = format(meas_state, f"0{nqubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + int(round(s.probability * shots_per_state))
        total = sum(counts.values())
        assert abs(total - shots_per_state) <= 1, \
            f"SPAM calibration counts for input {input_bitstring} sum to {total}, expected {shots_per_state}"
        counts_per_input[input_bitstring] = counts

    counts_per_input_per_chunk = {f"chunk_{chunk_id}": counts_per_input}
    # Save the dictionary using get_run_subfolder - append to it if a dictionary exists before and append is True
    path = get_run_subfolder(config, "raw_results", "spam_calibration_counts_per_chunk.json")
    if append_counts_dict and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(counts_per_input_per_chunk)
        to_save = existing
    else:
        to_save = counts_per_input_per_chunk

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=4)

    return counts_per_input_per_chunk


# ================================================ MERGING ACROSS CHUNKS + CONFUSION MATRIX ==================================================== #
def load_merged_spam_calibration_counts(config, n_samples=None):
    """
    Reads spam_calibration_counts_per_chunk.json (keyed by "chunk_0", "chunk_1", ...,
    merges the dictionaries for all chunks.

    :param n_samples: if given, sum only the first n_samples collected chunks (in chunk
                       order; "chunk_0".."chunk_{n_samples-1}" for a run without gaps,
                       see source/core/records/chunk_scope.py) instead of every chunk present — for
                       analyzing a subset of a run without touching the raw per-chunk
                       data. None = all chunks (unchanged default behaviour).
    """
    path = get_run_subfolder(config, "raw_results", "spam_calibration_counts_per_chunk.json")
    with open(path, "r", encoding="utf-8") as f:
        counts_per_chunk = json.load(f)

    if n_samples is not None:
        present = sorted((k for k in counts_per_chunk if k.startswith("chunk_")), key=lambda k: int(k.split("_")[1]))
        assert len(present) >= n_samples, (
            f"n_samples={n_samples}, but only {len(present)} chunks have SPAM calibration counts")
        counts_per_chunk = {k: counts_per_chunk[k] for k in present[:n_samples]}

    totals = {}
    for chunk_key, counts_per_input in counts_per_chunk.items():
        for input_bitstring, counts in counts_per_input.items():
            totals.setdefault(input_bitstring, {})
            for measured_bitstring, c in counts.items():
                totals[input_bitstring][measured_bitstring] = totals[input_bitstring].get(measured_bitstring, 0) + c

    return totals

def count_spam_calibration_chunks(config):
    """Number of chunks with SPAM calibration counts in raw_results/ (the maximum a SPAM correction can use)."""
    path = get_run_subfolder(config, "raw_results", "spam_calibration_counts_per_chunk.json")
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for k in json.load(f) if k.startswith("chunk_"))


def build_confusion_matrix_from_counts(totals, nqubits):
    """
    Builds the 2**nqubits x 2**nqubits SPAM confusion matrix M, where
        M[i, j] = P(measure i | prepared j)
    Each column j (one prepared basis state) is normalized to sum to 1.
    """
    dim = 2**nqubits
    M = np.zeros((dim, dim))
    for j in range(dim):
        input_bitstring = format(j, f"0{nqubits}b")
        counts = totals.get(input_bitstring, {})
        total  = sum(counts.values())
        assert total > 0, f"No calibration shots recorded for input state {input_bitstring}!"
        for meas_bitstring, c in counts.items():
            i = int(meas_bitstring, 2)
            M[i, j] = c / total
    return M


def compute_and_save_spam_confusion_matrix(config, n_samples=None):
    """
    Builds the confusion matrix:
    spam_confusion_matrix.npy / spam_confusion_matrix_inv.npy.

    :param n_samples: if given and different from config["samples"] (the full
                       run size), restricts calibration to the first n_samples
                       chunks and saves to spam_confusion_matrix[_inv]+nsamples-{n_samples}.npy
                       instead — the full run's canonical matrix files are never
                       touched. None, or equal to config["samples"], uses every
                       collected chunk and writes the unsuffixed files. The analysis
                       (mitigation.py) asks for a subset matrix only on an explicit
                       spam_n_samples; by default it uses the unsuffixed one.
    :returns: (M, M_inv)
    """
    nqubits  = config["nsites"]
    subset   = n_samples is not None and n_samples != config["samples"]
    totals   = load_merged_spam_calibration_counts(config, n_samples=n_samples if subset else None)

    M     = build_confusion_matrix_from_counts(totals, nqubits)
    M_inv = np.linalg.inv(M)

    suffix = subset_suffix(n_samples) if subset else ""
    np.save(get_run_subfolder(config, "processed_results", f"spam_confusion_matrix{suffix}.npy"),     M)
    np.save(get_run_subfolder(config, "processed_results", f"spam_confusion_matrix_inv{suffix}.npy"), M_inv)

    return M, M_inv

# ================================================ APPLYING THE CORRECTION ====================================================================== #
def apply_spam_correction_to_bitstring_counts(counts, M_inv, nqubits):
    """
    Applies the inverse-confusion-matrix SPAM correction to a single
    bitstring -> count dict.
    """
    dim = 2**nqubits
    raw_vec = np.zeros(dim)
    for bitstring, c in counts.items():
        raw_vec[int(bitstring, 2)] = c
 
    corrected_vec = M_inv @ raw_vec
 
    return {
        format(idx, f"0{nqubits}b"): float(corrected_vec[idx])
        for idx in range(dim)
    }
 
 
def apply_spam_correction_nested(nested_counts_dict, M_inv, nqubits):
    """
    Applies SPAM correction to every bitstring-count leaf dict inside a
    nested {job_key: {circuit_key: {bitstring: count}}} structure. Used for
    both `measurement_counts` (raw/rc/crc Trotter counts) and counts-based
    `fidelity_nec_dict` entries. Non-dict-shaped leaves (list-valued
    fidelity entries from the scalar-observable NEC mode) are left
    untouched, since those come from analytic expectation values rather
    than sampled counts.
    """
    corrected = {}
    for job_key, entry in nested_counts_dict.items():
        if not isinstance(entry, dict):
            warnings.warn(
                f"SPAM correction skipped for job {job_key!r}: entry is not a "
                f"bitstring-count dict (got {type(entry).__name__}), so no "
                f"readout mitigation can be performed on it. Passing it through unchanged."
            )
            corrected[job_key] = entry
            continue
        corrected[job_key] = {
            circuit_key: apply_spam_correction_to_bitstring_counts(counts, M_inv, nqubits)
            for circuit_key, counts in entry.items()
        }
    return corrected
