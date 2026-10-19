# myQLM is imported only inside the methods that build circuits, so this module
# (and every submit/collect/analysis script) loads without it — see source/core/common.py.
import numpy as np
import os
from source.core.common import *
from source.core.circuits.SPAM_functions import *
from source.core.analysis.mitigation import check_shots_consistency, merge_chunks, post_processing_and_saving
from source.core.circuits.building import build_BCS_time_evolution_circuit, build_preplugin, cache_exact_observables, calculate_infidelity_theta, calculate_thetas_from_infidelity, exact_observables_path, load_or_compile_batch, missing_exact_observables, prepare_initial_batch
from source.core.records.timestamps import TIMESTAMP_FILENAME, TIMESTAMP_FILENAME_V13, sync_circuit_timestamp_logs, timestamp
import json
import gc
import sys
import time
from tqdm import tqdm
from source.core.circuits.circuit_drawings import export_circuit_drawings, chunk_items
from source.core.records.run_info import write_run_info
from source.core.records.run_names import parse_run_name
from source.core.records.health_check_records import now_utc_iso, format_chunk_range
from source.core.records.job_records import (archive_failed_record, write_job_record, run_relative_path, saved_config,
                                     chunks_without_submission,
                                     MACHINE_SPECIFIC_CONFIG_KEYS)
from source.core.records.job_timing import (
    submit_timing, merge_timing, ibm_collect_timing, iqm_resonance_collect_timing, iqm_raw_job,
    eqe_collect_timing, aqt_timing, aqt_shot_parts, AQT_MAX_SHOTS,
)
from source.core.records import circuit_check as CC
from source.core.qpu import iqm_vlq as VLQ
from source.core.qpu.providers import job_state_of_status
from source.core.qpu.layout_selection import drop_identity_final_layout
from source.core.qpu.qpu_snapshot import scheduling_method
from source.core.qpu.aqt import _aqt_run, aqt_circuit_qpu_us
from source.core.qpu.remote_jobs import _failed_job_timing, _remote_job_state, _submit_error_kind, _warn_circuit_check, ChunkSubmitError, SUBMIT_RETRY_BACKOFF_S, print_job_line
import warnings
from source.core.records.run_names import IBM_CLOUD, IQM_RESONANCE, QAAS_IT4I, MQSS_LRZ, PCSS

# Circuits are built in local_sim.venv (Qiskit 2.x) and read by older-Qiskit venvs (e.g.
# aqt.venv, Qiskit 1.4.6) from QPY files written at a format version they support (all_to_all:
# QPY 13). Qiskit still warns on load because the header records the newer *writer* version —
# that is the intended mode here, so this one warning is silenced; a truly unreadable file
# still fails on load.
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"The qiskit version used to generate the provided QPY file")

# ── IQM job size: most circuits one job may hold ──────────────────────────────
# submit() packs as many whole chunks (each with its SPAM circuits) into one IQM job
# as fit under this limit. Resonance: 100 circuits per job is accepted, 101 is not.
# EQE1: 70 circuits per job is the limit communicated directly by the EQE1 staff.
IQM_MAX_CIRCUITS_PER_JOB = {IQM_RESONANCE: 100, MQSS_LRZ: 70}
# IQM VLQ (IT4I, via qaas): qaas uploads every circuit as its own HTTP request (~0.3 s each).
# Jobs of 1,000 circuits run; jobs of 3,264 circuits (48 full-benchmark chunks) upload, but then
# fail on IT4I's side when HEAppE forwards them to IQM ("The write operation timed out",
# 2026-09-29), and each keeps one upload open for ~16–18 min, where a network drop loses the
# whole job. A job therefore holds VLQ_CHUNKS_PER_JOB whole chunks (or
# submit(chunks_per_qpu_job_batch=...)): 12 full-benchmark chunks are 816 circuits, ~4 min to upload.
VLQ_CHUNKS_PER_JOB = 12
# IBM Batch jobs hold this many whole chunks: the full benchmark's 480 chunks go out as 10 jobs.
IBM_CHUNKS_PER_JOB = 48


def _chunk_in(path, key):
    """True when the JSON file at path exists, is readable and holds the key (e.g. "chunk_3")."""
    try:
        return path.exists() and key in json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False


class bacq_ps_qbcs:
    def __init__(
            self, infidelity_total=0.0, ratio_ct=0.0, noise_type="pauli",
            folder_prefix="results", nshots=10, sample_size=480,   # folder_prefix: unused, accepted for older scripts
            chunk_size=None, list_circuit_times=None,
            infidelity_from_counts=True, nec_random_initial_state=True,
            do_SPAM=False, SPAM_shots_per_state=10, dt=0.2,
            error_mitigation_strategies = None, folder_path=None,   # folder_path: unused, accepted for older scripts
            rng_seed=0, verbose=False
            ):
        
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
        if list_circuit_times is None: 
            list_circuit_times = list(np.arange(0.2, 4.0, 0.4))        

        if error_mitigation_strategies is None:
            error_mitigation_strategies = (["SPAM"] if do_SPAM else []) + ["NEC", "RC+NEC", "cRC+NEC"]
        assert do_SPAM == ("SPAM" in error_mitigation_strategies), \
        "Inconsistency: do_SPAM flag and error_mitigation_strategies disagree on SPAM."

        CONFIG = {
            "noise_val": cnot_noise_angle,
            "neighbor_noise_val": neighbor_noise_angle,
            "nsites": 3,
            "epsilon": np.array([-1, 0, 1]),
            "g": 0.5,
            "T_list": list_circuit_times,
            "ind_T": None, 
            "dt": dt,
            "samples": sample_size, #512,
            "split_raw+nec_jobs": True, #CHANGE BEFORE RUNNING
            "nec_measurement_counts": infidelity_from_counts, #CHANGE BEFORE RUNNING
            "nec_randomize_input_state": nec_random_initial_state,
            "chunk_size": chunk_size, # None for no chunking
            "obs_str": "XYZ",
            "rng_seed": rng_seed,
            "cnots_per_trotter": 6,
            "local_qubit_basis": ["X", "Y", "Z"],
            "shots" : int(nshots),
            "SPAM_correction": do_SPAM,
            "SPAM_circuit_shots_per_input_state": SPAM_shots_per_state,
            "lnn_architecture": False,
            # Coherent noise is compiled into the cached circuits, so it names its own cache (get_circuit_folder()).
            "coherent_noise": (dict(noise_type=noise_type, infidelity_total=float(infidelity_total), ratio_ct=float(ratio_ct))
                               if infidelity_total > 1e-8 else None),
        }
        CONFIG["T"]                     = None if CONFIG["ind_T"] is None else CONFIG["T_list"][CONFIG["ind_T"]]
        CONFIG["Nt"]                    = int(round(CONFIG["T"] / CONFIG["dt"])) if CONFIG["T"] is not None else None

        if noise_type=="depolarizing":
            CONFIG["noise_angles"]          = np.array([1,1,1,1,1,1]) * CONFIG["noise_val"] # np.ones(6) * CONFIG["noise_val"]
            CONFIG["neighbor_noise_angles"] = np.array([1,1,1]) * CONFIG["neighbor_noise_val"] # np.ones(3) * CONFIG["neighbor_noise_val"]
        elif noise_type=="pauli":
            CONFIG["noise_angles"]          = np.array([1,0,0,1,0,0]) * CONFIG["noise_val"] # np.ones(6) * CONFIG["noise_val"]
            CONFIG["neighbor_noise_angles"] = np.array([1,0,0]) * CONFIG["neighbor_noise_val"] # np.ones(3) * CONFIG["neighbor_noise_val"]


        assert len(CONFIG["epsilon"])==CONFIG["nsites"]
        assert len(CONFIG["local_qubit_basis"])==CONFIG["nsites"]

        self.TIMESTAMP_FILE = None 

        self.ems_to_job_name = {"NEC": "raw", "RC+NEC": "rc", "cRC+NEC": "crc"}
        nec_ems_strategies  = [e for e in error_mitigation_strategies if e != "SPAM"]

        self.CONFIG                      = CONFIG
        self.error_mitigation_strategies = error_mitigation_strategies # "NEC" creates two jobs - [noisy_job, noisy_NEC], "RC+NEC" creates 2*Nsample jobs - [RC jobs, RC-NEC jobs], , "cRC+NEC" creates 2*Nsample jobs - [RC jobs, RC-NEC jobs]
        self.nec_ems_strategies          = nec_ems_strategies
        self.job_names                   = [self.ems_to_job_name[ems] for ems in nec_ems_strategies]
        self.roles                       = [tag for name in self.job_names for tag in (name, f"{name}_nec")] # ["raw", "raw_nec", "rc", "rc_nec", "crc", "crc_nec"]
        self.trotter_steps_list          = np.round(np.array(CONFIG["T_list"]) / CONFIG["dt"]).astype(int)
        self.job_tags                    = [f"Nt_{num_trotter_steps:02d}_{tag}" for num_trotter_steps in self.trotter_steps_list for tag in self.roles]
        self.n_depths                    = len(self.trotter_steps_list)
        self.num_total_jobs              = len(CONFIG["T_list"])*(len(self.roles))*CONFIG["samples"]
        self._circuits_rotated           = False
        self._status                     = "initialised"
        self._fake_job_registry          = {}
        self._verbose                    = verbose   # gates per-chunk/per-sample detail prints; progress bars show regardless
        self._qiskit_v13_qpy             = False   # set by build(): True for all_to_all circuits
        self.run_instance                = (None, None)   # IBM: (instance name, plan) of the run, recorded in run_info.json; set by submit_async_run()
        self._last_transpile             = {"transpile_s": None, "transpile_reused": False}   # set by transpile_chunk_for_backend_to_isa()

        suggested_chunk_size = len(self.roles) * self.n_depths
        if CONFIG["chunk_size"] != suggested_chunk_size:
            print(f"⚠️  chunk_size={CONFIG['chunk_size']} is inconsistent with the other variables "
                  f"(suggested = len(roles)*len(T_list) = {suggested_chunk_size}) — using {suggested_chunk_size} instead.")
            CONFIG["chunk_size"] = suggested_chunk_size
        assert CONFIG["chunk_size"]==len(self.roles)*self.n_depths, f"Chunk size must be {len(self.roles)*self.n_depths}!!"
    
    def build_qbcs_circuits(self, CONFIG=None, circuit_transpilation_LNN=False, rotate_qubits_locally=False):
        from qat.core import Batch
        if CONFIG is None:
            CONFIG = self.CONFIG
        self.CONFIG["lnn_architecture"] = circuit_transpilation_LNN

        circuit_folder  = get_circuit_folder(self.CONFIG, top_folder=True)
        target_filename = TIMESTAMP_FILENAME_V13 if self._qiskit_v13_qpy else TIMESTAMP_FILENAME
        # Backfill shared-stage rows from the other chunk format's log (if any; local-time and
        # UTC logs separately) before this build's own stages start deciding what to skip —
        # so both logs stay complete whichever format is built first.
        sync_circuit_timestamp_logs(circuit_folder, self._qiskit_v13_qpy)
        self.TIMESTAMP_FILE = Path(circuit_folder) / target_filename

        # ── Idempotency: reuse the cached circuit batch if it already exists ───
        batch_path = os.path.join(circuit_folder, "rotated_circuit_batch.batch")
        if os.path.exists(batch_path):
            self.circuit_batch     = Batch.load(batch_path)
            assert len(self.circuit_batch.jobs) == self.n_depths, (
                f"Cached circuit batch at {batch_path} has "
                f"{len(self.circuit_batch.jobs)} depths, but current config expects "
                f"{self.n_depths} — stale/mismatched cache."
            )
            self._circuits_rotated = rotate_qubits_locally
            self._use_circuit_LNN  = circuit_transpilation_LNN
            self._reset_config_depth()
            return self.circuit_batch

        circuit_jobs = []
        for ti, t_circuit in enumerate(CONFIG["T_list"]):
            CONFIG["ind_T"]   = ti   
            CONFIG["T"]       = t_circuit
            num_trotter_steps = int(round(t_circuit/CONFIG["dt"]))
            CONFIG["Nt"]      = num_trotter_steps
            timestamp(label=f"Starting compilation for Nt={num_trotter_steps}.", timestamp_file=self.TIMESTAMP_FILE)
            # ── 1. Build circuits ─────────────────────────────────────────────
            circuit_kwargs = {k: CONFIG[k] for k in ["nsites", "g", "epsilon", "T", "dt", "local_qubit_basis"]}
            circuit_original, rotated_circuit = build_BCS_time_evolution_circuit(**circuit_kwargs, LNN=circuit_transpilation_LNN)

            if not rotate_qubits_locally: 
                circuit_jobs.append(circuit_original.to_job())
            else: 
                circuit_jobs.append(rotated_circuit.to_job())
        
        self.circuit_batch = Batch(circuit_jobs)
        self.circuit_batch.dump(batch_path)
        print(f"Saved circuit batch ({len(circuit_jobs)} depths) → {rel_path(batch_path)}")
        self._circuits_rotated = rotate_qubits_locally
        self._use_circuit_LNN  = circuit_transpilation_LNN
        
        self._reset_config_depth()
        return self.circuit_batch

    def build_nec_ems_job_batch(self, rotated_circuit_job_batch, ems, compute_nec_map=False):
        assert self._circuits_rotated, (
            "build_qbcs_circuits must be called with "
            "rotate_qubits_locally=True before build_nec_ems_job_batch.")

        n_depths = len(rotated_circuit_job_batch.jobs) # batch of the list of bare circuits for all depths — validated against self.n_depths in build_qbcs_circuits()

        if ems != "SPAM" and ems in self.nec_ems_strategies:
            for nti in range(n_depths):
                rotated_circuit  = rotated_circuit_job_batch.jobs[nti].circuit
                num_trotter_steps = self.trotter_steps_list[nti]

                # Update CONFIG for this depth
                self.CONFIG["ind_T"] = nti
                self.CONFIG["T"]     = self.CONFIG["T_list"][nti]
                self.CONFIG["Nt"]    = int(num_trotter_steps)

                assert int(round(self.CONFIG["T"]/self.CONFIG["dt"])) == self.CONFIG["Nt"]

                # ── Idempotency: skip this depth entirely (no compute, no timestamps)
                # if its batch is already cached ──────────────────────────────────
                nsamples   = self.CONFIG["samples"]
                batch_file = os.path.join(
                    get_circuit_folder(self.CONFIG, top_folder=False),
                    f"batch_jobs_Nt{self.CONFIG['Nt']}_{ems}_sample_{nsamples-1}.batch"
                )
                if os.path.exists(batch_file):
                    continue

                timestamp(
                    label          = f"Building {ems} batch for Nt={num_trotter_steps}.",
                    timestamp_file = self.TIMESTAMP_FILE
                )
                # Build plugins
                nec_plugin, rccnot_plugin = build_preplugin(
                    rotated_circuit,
                    self.CONFIG,
                    error_mitigation          = ems,
                    compute_classical_nec_map = compute_nec_map, # Need this only once per depth
                    verbose                   = self._verbose
                )

                # Compile batch
                batch      = prepare_initial_batch(
                    rotated_circuit,
                    shots    = self.CONFIG["shots"],
                    obs      = None,
                    job_type = self.ems_to_job_name[ems]
                )
                batch = load_or_compile_batch(nec_plugin,    batch, None,       verbose=self._verbose)
                batch = load_or_compile_batch(rccnot_plugin, batch, batch_file, verbose=self._verbose)

                timestamp(
                    label          = f"Finished {ems} batch for Nt={num_trotter_steps}.",
                    timestamp_file = self.TIMESTAMP_FILE
                )
            print(f"[{ems}] All {n_depths} depth batches compiled and saved.")
            self._reset_config_depth()
        else:
            raise ValueError(f"Invalid or unsupported EMS in "
                            f"build_nec_ems_job_batch: '{ems}'. "
                            f"SPAM is handled separately.")
        
        


    def build_all_nec_ems_job_batches(self):
        assert hasattr(self, "circuit_batch") and self.circuit_batch is not None and len(self.circuit_batch.jobs) > 0, (
            "circuit_batch is missing or empty. "
            "Call build_qbcs_circuits(rotate_qubits_locally=True) first."
        )
        for im, ems in enumerate(self.nec_ems_strategies):
            self.build_nec_ems_job_batch(
                rotated_circuit_job_batch = self.circuit_batch,
                ems                       = ems,
                compute_nec_map           = (im == 0)
            )

    def compute_chunk_arithmetic(self):
        num_total_jobs = self.n_depths * len(self.roles) * self.CONFIG["samples"]

        if self.CONFIG["chunk_size"] is not None:
            chunk_size = self.CONFIG["chunk_size"]
        else:
            chunk_size = num_total_jobs

        n_chunks                 = int(round(num_total_jobs / chunk_size))
        rc_samples_per_ems_chunk = int(round(chunk_size / (self.n_depths * len(self.roles)))) # = 1

        assert n_chunks * chunk_size == num_total_jobs, \
            f"n_chunks ({n_chunks}) × chunk_size ({chunk_size}) ≠ num_total_jobs ({num_total_jobs})."
        assert rc_samples_per_ems_chunk == 1, \
            "Each chunk has exactly one sample of an EMS"
        assert self.num_total_jobs == num_total_jobs, \
            "num_total_jobs is initialized twice with different values"
        
        self.chunk_size               = chunk_size
        self.n_chunks                 = n_chunks
        self.rc_samples_per_ems_chunk = rc_samples_per_ems_chunk

        print(f"num_total_jobs={num_total_jobs}, chunk_size={chunk_size}, "
              f"n_chunks={n_chunks}, rc_samples_per_ems_chunk={rc_samples_per_ems_chunk}")


    def interleave_job_batches(self):
        """myQLM interleaving only — chunks the per-depth/per-ems batches into
        batch_jobs_interleaved_chunk{nc}.batch files. Independent of, and
        cached separately from, convert_chunks_to_qiskit()'s QPY dump."""
        assert hasattr(self, "n_chunks"), (
            "compute_chunk_arithmetic() must be called before interleave_job_batches()."
        )

        circ_top = get_circuit_folder(self.CONFIG, top_folder=True)
        migrate_flat_chunk_files(circ_top)
        myqlm_chunk_folder = get_chunk_folder(circ_top, "myqlm")
        expected_paths = [
            os.path.join(myqlm_chunk_folder, f"batch_jobs_interleaved_chunk{nc}")
            for nc in range(self.n_chunks)
        ]

        # ── Idempotency: skip entirely (no compute, no timestamps) if every
        # chunk is already interleaved ──────────────────────────────────────
        if all(os.path.exists(f"{p}.batch") for p in expected_paths):
            self.chunk_paths = expected_paths
            return

        from source.core.circuits.main import interleave_the_batch_multidepth_from_disk
        batch_names = list(self.nec_ems_strategies)

        timestamp(label="Starting myQLM interleaving.", timestamp_file=self.TIMESTAMP_FILE)

        self.chunk_paths = interleave_the_batch_multidepth_from_disk(
            config       = self.CONFIG,
            t_list       = self.CONFIG["T_list"],
            n_rc_samples = self.CONFIG["samples"],
            batch_names  = batch_names,
            verbose      = self._verbose
        )

        assert len(self.chunk_paths) == self.n_chunks, (
            f"Expected {self.n_chunks} chunk paths, "
            f"got {len(self.chunk_paths)}."
        )

        timestamp(
            label          = f"Finished myQLM interleaving {self.n_chunks} chunks.",
            timestamp_file = self.TIMESTAMP_FILE
        )

    def convert_chunks_to_qiskit(self):
        """Dump each interleaved chunk to Qiskit QPY. Saves chunk_{nc}_qiskit.qpy
        at Qiskit's default QPY version for LNN, or chunk_{nc}_qiskit_v13.qpy at QPY
        version 13 for all_to_all (self._qiskit_v13_qpy, set by build()) — the
        version the older Qiskit of aqt.venv and iqm_vlq.venv can read. Independent
        of, and cached separately from, interleave_job_batches()'s myQLM stage."""
        assert hasattr(self, "chunk_paths"), (
            "interleave_job_batches() must be called before convert_chunks_to_qiskit()."
        )

        kind        = "qiskit_v13" if self._qiskit_v13_qpy else "qiskit"
        suffix      = kind
        circ_top    = get_circuit_folder(self.CONFIG, top_folder=True)
        migrate_flat_chunk_files(circ_top)
        circ_folder = Path(get_chunk_folder(circ_top, kind))
        qpy_paths   = [circ_folder / f"chunk_{nc}_{suffix}.qpy" for nc in range(self.n_chunks)]

        # ── Idempotency: skip entirely (no compute, no timestamps) if every
        # chunk is already converted to this format ─────────────────────────
        if all(p.exists() for p in qpy_paths):
            return

        timestamp(label="Starting Qiskit conversion of chunks.", timestamp_file=self.TIMESTAMP_FILE)

        from qiskit.qpy import dump as qpy_dump
        qpy_version = 13 if self._qiskit_v13_qpy else None
        for nc, qpy_path in enumerate(tqdm(qpy_paths, desc="Converting chunks to Qiskit", mininterval=0.5)):
            if not qpy_path.exists():
                qk_circuits = self.convert_myqlm_chunk_to_qiskit(nc)
                with open(qpy_path, "wb") as f:
                    if qpy_version is not None:
                        qpy_dump(qk_circuits, f, version=qpy_version)
                    else:
                        qpy_dump(qk_circuits, f)
                if self._verbose: print(f"  [QPY] Saved chunk {nc} → {rel_path(qpy_path)}")

        timestamp(
            label          = f"Finished Qiskit conversion {self.n_chunks} chunks.",
            timestamp_file = self.TIMESTAMP_FILE
        )

    def _cleanup_empty_circuit_folders(self):
        """
        Remove empty leftover folders from compiled_from_plugins/ — a side effect
        of RCCNOT_plugin/cRCCNOT_plugin.save_data() creating an (unused) exp_dir
        on every twirled-role compile() call (see USER_MANUAL §4.1). Only ever deletes
        a folder that is actually empty (os.rmdir fails otherwise), so this is
        safe even if something unexpected populates one of these in the future,
        or another concurrent build() on the same shared topology folder is
        mid-flight.
        """
        circ_dir = Path(get_circuit_folder(self.CONFIG, top_folder=False))
        if not circ_dir.is_dir():
            return
        removed  = 0
        for entry in circ_dir.iterdir():
            if entry.is_dir():
                try:
                    entry.rmdir()
                    removed += 1
                except OSError:
                    pass   # not empty (or not ours) — leave it alone
        if removed:
            print(f"  [cleanup] Removed {removed} empty leftover folder(s) from {rel_path(circ_dir)}.")

    def convert_myqlm_chunk_to_qiskit(self, nc):
        require_myqlm(f"Converting chunk {nc} from myQLM (its Qiskit QPY file is missing)")
        from qat.core import Batch
        from qat.qiskit.converters import QaptivaToQiskit
        from qiskit.circuit.barrier import Barrier
        

        assert hasattr(self, "chunk_paths"), (
            "interleave_job_batches() must be called before convert_myqlm_chunk_to_qiskit()."
        )
        assert nc < len(self.chunk_paths), (
            f"Chunk index {nc} out of range — only {len(self.chunk_paths)} chunks available."
        )
        gates_dict = {
            "BARRIER": lambda num_qubits=self.CONFIG["nsites"]: Barrier(num_qubits)
        }
        chunk_batch  = Batch.load(self.chunk_paths[nc])
        translator   = QaptivaToQiskit(gates_dict=gates_dict)

        qk_circuits  = [translator.to_circ(job.circuit) for job in chunk_batch.jobs]

        del chunk_batch
        gc.collect()

        if self._verbose: print(f"  [Chunk {nc}] Converted {len(qk_circuits)} myQLM circuits to Qiskit.")
        return qk_circuits


    def _chunk_results_complete(self, nc):
        """True if the run's raw results already hold all of chunk nc's counts, NEC
        fidelities and (with SPAM) its SPAM counts."""
        n_nec_ems = len(self.nec_ems_strategies)
        key       = f"chunk_{nc}"

        def _has(fname, n):
            path = get_run_subfolder(self.CONFIG, "raw_results", fname)
            data = load_json(path) if os.path.exists(path) else {}
            return key in data and len(data[key]) == n

        return (_has("results_counts.json",   self.n_depths * n_nec_ems)     # non-NEC roles per chunk
                and _has("results_fidelity.json", self.n_depths * n_nec_ems)   # NEC roles per chunk
                and (not self.CONFIG["SPAM_correction"]
                     or _has("spam_calibration_counts_per_chunk.json", 2**self.CONFIG["nsites"])))

    def chunks_to_submit(self, run_folder, max_chunks=None):
        """The chunks of [0, max_chunks) (all chunks if None) still to be submitted: those
        without a job record with status 'submitted' / 'complete' (real QPUs), or, for a
        local simulator run (a simulation data type), those without complete results — its job records are
        written together with the results, so the results decide (a 'complete' record whose
        results were deleted is simulated again). A submit script calls this once and passes the
        list to submit(chunks=...) and to its health checks (submit_health_check(chunks=...))."""
        if self.CONFIG.get("folder_path") is None:
            self.CONFIG["folder_path"] = os.path.join(PROJECT_ROOT, "datafiles", run_folder)
        n = self.n_chunks if max_chunks is None else min(max_chunks, self.n_chunks)
        if parse_run_name(self.CONFIG["folder_path"]).is_simulation:
            return [nc for nc in range(n) if not self._chunk_results_complete(nc)]
        return [nc for nc in chunks_without_submission(self.CONFIG["folder_path"], n)
                if not self._chunk_results_complete(nc)]

    def submit_chunk_local_qiskit(self, nc, qiskit_sampler, backend=None):
        """
        Submit a single chunk to a Qiskit sampler and save results to disk.

        :param nc:              int — chunk index (starts from 0)
        :param qiskit_sampler:  pre-constructed SamplerV2 object (AerSamplerV2 or IBM SamplerV2)
        """
        from qiskit.qpy import load

        chunk_key           = f"chunk_{nc}"
        counts_path         = get_run_subfolder(self.CONFIG, "raw_results", "results_counts.json")
        fidelity_path       = get_run_subfolder(self.CONFIG, "raw_results", "results_fidelity.json")

        assert self.chunk_size==self.n_depths*len(self.roles), "chunk_size is not consistent!!"
        if self.CONFIG["SPAM_correction"]:
            assert hasattr(self, "_qk_spam_circuits"), (
                "call build_spam_batch_qiskit() before submit_chunk() when SPAM_correction=True."
            )
            n_spam = len(self._qk_spam_circuits)
        else:
            n_spam = 0

        # ── Idempotency check ─────────────────────────────────────────────────────
        if self._chunk_results_complete(nc):
            print(f"  [Chunk {nc}] Already complete — skipping.")
            return False
        existing_counts   = load_json(counts_path)   if os.path.exists(counts_path)   else {}
        existing_fidelity = load_json(fidelity_path) if os.path.exists(fidelity_path) else {}
        if self.CONFIG["SPAM_correction"]:
            spam_path     = get_run_subfolder(self.CONFIG, "raw_results", "spam_calibration_counts_per_chunk.json")
            existing_spam = load_json(spam_path) if os.path.exists(spam_path) else {}

        # ── Step 1: Convert chunk to Qiskit circuits + add measurements ───────────
        timestamp(label=f"Chunk {nc}: converting to Qiskit.", timestamp_file=self.TIMESTAMP_FILE)

        self._last_transpile = {"transpile_s": None, "transpile_reused": False}
        if backend is not None:
            qk_circuits = self.transpile_chunk_for_backend_to_isa(nc, backend)
        else:
            qk_circuits = self._load_raw_chunk_qiskit(nc)   # cached QPY; myQLM only if it's missing
            spam_items  = ([("00_SPAM", self._qk_spam_circuits[0], None)]
                           if self.CONFIG["SPAM_correction"] and getattr(self, "_qk_spam_circuits", None) else [])
            export_circuit_drawings(self.CONFIG["folder_path"], spam_items + chunk_items(self.roles, qk_circuits))

        if self.CONFIG["SPAM_correction"]:
            if backend is not None:
                spam_isa_path = Path(get_run_subfolder(self.CONFIG, "transpiled_circuits")) / "spam_transpiled.qpy"
                if not spam_isa_path.exists():
                    self.transpile_spam_for_backend_to_isa(backend)
                with open(spam_isa_path, "rb") as f:
                    spam_isa = load(f)
                all_circuits = spam_isa + qk_circuits
            else:
                all_circuits = self._qk_spam_circuits + qk_circuits
        else:
            all_circuits = qk_circuits

        # ── Step 2: Run on sampler (local simulation: run + result is the CPU-side "submission") ──
        timestamp(label=f"Chunk {nc}: submitting to backend.", timestamp_file=self.TIMESTAMP_FILE)
        record_path  = Path(get_run_subfolder(self.CONFIG, "job_records")) / f"job_record_chunk_{nc}.json"
        record       = self._local_job_record(nc, qiskit_sampler, backend, n_spam, len(qk_circuits),
                                              submitted_at=now_utc_iso(precise=True))
        sim_t0       = time.perf_counter()
        try:
            result, submit_s = self._timed_submit(
                f"chunk{nc}", lambda: qiskit_sampler.run(all_circuits, shots=self.CONFIG["shots"]).result(),
                retry=False)
        except BaseException as e:   # Ctrl+C too: the attempt is on record, then archived when the chunk reruns
            record.update(status="failed", failure=scrub_paths(f"{type(e).__name__}: {e}")[:1000], failure_stage="execution",
                          may_exist_on_provider=False, finished_at=now_utc_iso(precise=True))
            merge_timing(record, submit_s=round(time.perf_counter() - sim_t0, 3))
            archive_failed_record(record_path.parent, nc)
            write_job_record(record_path, record)
            print(f"  ❌ [Chunk {nc}] Simulation failed — recorded as failed (job_records/).", file=sys.stderr)
            raise
        record["finished_at"] = now_utc_iso(precise=True)
        collect_t0 = time.perf_counter()
        self._log_timestamp(f"chunk{nc}_collect_start")   # handling of the received results: the chunk's collection
        spam_counts = [result[si].data.meas.get_counts() for si in range(n_spam)]
        all_counts  = [result[ci].data.meas.get_counts() for ci in range(n_spam, n_spam + len(qk_circuits))]
        timestamp(label=f"Chunk {nc}: results received.", timestamp_file=self.TIMESTAMP_FILE)

        # ── Step 3: Organise into key structure ───────────────────────────────────
        measurement_counts = {}
        fidelity_nec_dict  = {}


        for di, nt in enumerate(self.trotter_steps_list):
            tag = f"Nt_{nt:02d}"
            for ri, role in enumerate(self.roles):
                # local_idx: ranges 0 to n_depths * len(self.roles) - 1
                # identical across all chunks — same keys appear in every chunk_{nc}
                local_idx = di * len(self.roles) + ri
                job_key   = f"job_{tag}_{role}_ems{local_idx}"
                circ_key  = f"{tag}_{role}_circuit_0"   # always 0 — sample index inside a chunk (1 sample per chunk)
                counts    = {k[::-1]: v for k, v in all_counts[local_idx].items()}

                if role.endswith("_nec"):
                    fidelity_nec_dict[job_key]  = {circ_key: counts}
                else:
                    measurement_counts[job_key] = {circ_key: counts}

        # ── Step 3b: Save SPAM counts ─────────────────────────────────────────────
        if self.CONFIG["SPAM_correction"]:
            spam_counts_per_input = {}
            for si in range(n_spam):
                input_bitstring = format(si, f"0{self.CONFIG['nsites']}b")  # "000", "001", ..., "111"
                spam_counts_per_input[input_bitstring] = {
                    k[::-1]: v for k, v in spam_counts[si].items()  # reverse bitstring: Qiskit → myQLM ordering
                }

            # Append to spam_calibration_counts_per_chunk.json
            # spam_path                = get_exp_file(self.CONFIG, "spam_calibration_counts_per_chunk.json", top_folder=True)
            # existing_spam            = load_json(spam_path) if os.path.exists(spam_path) else {}
            existing_spam[chunk_key] = spam_counts_per_input
            with open(spam_path, "w", encoding="utf-8") as f:
                json.dump(existing_spam, f, indent=4)

        # ── Step 4: Save to disk ──────────────────────────────────────────────────
        existing_counts[chunk_key]   = measurement_counts
        existing_fidelity[chunk_key] = fidelity_nec_dict

        with open(counts_path, "w", encoding="utf-8") as f:
            json.dump(existing_counts, f, indent=4)
        with open(fidelity_path, "w", encoding="utf-8") as f:
            json.dump(existing_fidelity, f, indent=4)

        # ── Step 5: Job record, as for a real QPU (status 'complete': the results are saved) ──
        record["status"] = "complete"
        merge_timing(record, submit_s=submit_s, collect_s=round(time.perf_counter() - collect_t0, 3))
        archive_failed_record(record_path.parent, nc)   # keep a failed attempt's record
        write_job_record(record_path, record)
        self._log_timestamp(f"chunk{nc}_collect_end")

        print_chunk_saved(nc, n_spam if self.CONFIG["SPAM_correction"] else 0, len(measurement_counts), len(fidelity_nec_dict))

        del qk_circuits, all_counts
        gc.collect()
        return True

    def _local_job_record(self, nc, sampler, layout_backend, n_spam, n_circuits, submitted_at):
        """The job record of a locally simulated chunk — the fields of a real QPU's record that
        apply to a simulation (no job id, queue or QPU time; the simulator returns no executed
        circuits). submit_chunk_local_qiskit() completes it with the status and timing."""
        pq = getattr(self, "_physical_qubits", None) if layout_backend is not None else None
        return {
            "chunk_id":                nc,
            "status":                  "submitted",
            "service_name":            "local",
            "backend_name":            getattr(layout_backend, "name", None) or type(sampler).__name__,
            "sampler":                 type(sampler).__name__,
            "job_id":                  "N/A",
            "layout":                  {str(v): p for v, p in enumerate(pq)} if pq else "N/A",
            "submitted_at":            submitted_at,
            "scheduled_at":            None,
            "finished_at":             None,
            "nshots":                  self.CONFIG["shots"],
            "n_spam":                  n_spam,
            "n_circuits":              n_circuits,
            "submitted_circuit_files": self._submitted_circuit_files(nc, n_spam) if layout_backend is not None else [],
            "timing":                  submit_timing(**self._last_transpile),
            "circuit_check":           CC.not_available("local simulator: it does not return the executed circuits"),
            "result_path":             run_relative_path(self.CONFIG["folder_path"], "raw_results", "results_counts.json"),
            "trotter_steps_list":      [int(nt) for nt in self.trotter_steps_list],
            "roles":                   list(self.roles),
            "nsites":                  self.CONFIG["nsites"],
            "CONFIG": saved_config(self.CONFIG),
        }

    def submit_chunk_ibm(self, nc, sampler, backend, service=None):
        """
        Transpile chunk nc to ISA, submit asynchronously to IBM backend.
        Saves job_id to job_records/ and returns immediately — does NOT wait for results.
        Call get_results_by_chunk_ibm(exp_results_folder_path, nc) separately to collect results.

        Idempotent — skips if job record already exists for this chunk.

        :param nc:           int — chunk index
        :param backend:      IBM backend object (real or fake)
        :param service:      QiskitRuntimeService — required for real IBM, None for fake
        :param sampler:  qiskit_ibm_runtime.SamplerV2 — created once in submit() with correct mode (Batch or direct)
        """
        from qiskit.qpy import load
        import time

        record_path = Path(get_run_subfolder(self.CONFIG, "job_records")) / f"job_record_chunk_{nc}.json"

        # ── Idempotency ───────────────────────────────────────────────────────
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record['status'] in ("submitted", "complete"):
                print(f"  [Chunk {nc}] Already submitted (status={record['status']}) — skipping.")
                return

        # ── Transpile circuits ────────────────────────────────────────────────
        isa_circuits = self.transpile_chunk_for_backend_to_isa(nc, backend)

        # ── Print layout info (first circuit only — all share the same layout) ─
        from source.core.qpu.layout_selection import final_physical_qubits
        print(f"  [Chunk {nc}] Circuit 0: "
              f"depth={isa_circuits[0].depth()}, "
              f"gates={dict(isa_circuits[0].count_ops())}, "
              f"layout={final_physical_qubits(isa_circuits[0])}")

        # ── Add SPAM circuits if enabled ──────────────────────────────────────
        if self.CONFIG["SPAM_correction"]:
            spam_qpy = Path(get_run_subfolder(self.CONFIG, "transpiled_circuits")) / "spam_transpiled.qpy"
            assert spam_qpy.exists(), "Call transpile_spam_for_backend_to_isa(backend) before submit_chunk_ibm()."
            with open(spam_qpy, "rb") as f:
                spam_isa = load(f)
            all_circuits = spam_isa + isa_circuits
            n_spam       = len(spam_isa)
            assert len(spam_isa[0].cregs) > 0,   "SPAM ISA circuits have no classical registers!"
            assert len(isa_circuits[0].cregs) > 0, "ISA circuits have no classical registers!"
        else:
            all_circuits = isa_circuits
            n_spam       = 0
            assert len(isa_circuits[0].cregs) > 0, "ISA circuits have no classical registers!"

        # ── Submit ────────────────────────────────────────────────────────────
        transpiled    = dict(self._last_transpile)
        try:
            job, submit_s = self._timed_submit(f"chunk{nc}", lambda: sampler.run(all_circuits, shots=self.CONFIG["shots"]))
        except ChunkSubmitError as e:
            self._write_submission_failed_records([nc], IBM_CLOUD, backend, e, {nc: transpiled}, n_spam, len(isa_circuits))
            raise
        job_id = job.job_id()

        # ── Save job record ───────────────────────────────────────────────────
        record = {
            "backend_name": str(backend),
            "service_name": IBM_CLOUD,
            "chunk_id":     nc,
            "job_id":       job_id,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_at":  "N/A",
            "nshots":       self.CONFIG["shots"],
            "n_spam":       n_spam,
            "n_circuits":   len(isa_circuits),
            "submitted_circuit_files": self._submitted_circuit_files(nc, n_spam),
            "timing":       submit_timing(**transpiled, submit_s=submit_s),
            "submit_attempts": self._last_submit_attempts,
            "status":       "submitted",
            "result_path":  None,
            "trotter_steps_list": [int(x) for x in self.trotter_steps_list],
            "roles":        self.roles,
            "nsites":       self.CONFIG["nsites"],
            "CONFIG": saved_config(self.CONFIG)
        }
        archive_failed_record(record_path.parent, nc)   # keep a failed attempt's record (and its cost)
        write_job_record(record_path, record)
        print(f"  [Chunk {nc}] Submitted — job_id: {job_id}")
        print(f"  [Chunk {nc}] Record saved → {rel_path(record_path)}")

        # ── Store in fake registry if no real service ─────────────────────────
        if service is None:
            self._fake_job_registry[job_id] = job

    def submit(
        self,
        run_folder,          # str — subfolder under datafiles/, e.g. "run_01"
        layout_connection        = "all_to_all",    # fallback if build() wasn't called in this session
        lang                     = "qiskit", # "qiskit" | "myqlm"
        service                  = "local",  # "local" | "IBM_Cloud" | "MQSS_LRZ" | "IQM_Resonance" | "QAAS_IT4I" | "PCSS"
        backend                  = None,     # AerSamplerV2(), a provider backend object, etc.
        layout_backend           = None,      # Relevant for transpilation for the case of local qiskit noisy model runs,
        max_chunks               = None,      # only submits first max_chunks chunks if not None — async services only (IBM, IQM_*)
        chunks_per_qpu_job_batch = None,
        chunk_index              = None,      # synchronous real-QPU services only (AQT_PIAST-Q): submit exactly this one chunk
        physical_qubits          = None,      # optional list[int] — pin the run to these qubits instead of scoring (see resolve_physical_qubits)
        max_circuits_per_job     = None,      # IQM_* only: circuit limit of one packed job (None → IQM_MAX_CIRCUITS_PER_JOB)
        chunks                   = None,      # list[int] — submit exactly these chunks (from chunks_to_submit()); None: all left
        ):
        """
        Submit all interleaved chunks to the target backend sequentially.

        Sets the run folder and hardware connectivity if not already established
        in the self.CONFIG dictionary, then dispatches to the
        appropriate backend-specific submission method. For the real-QPU services,
        pass a connected backend object — the scripts in
        source/benchmark_scripts/real_qpu_scripts/ create it with the
        provider's token from tokens/ (source/core/qpu/tokens.py: read_token()).

        :param run_folder:         str — output subfolder under datafiles/, e.g. "run_01" or "test00_myqlm".
                                   Used to construct all result paths (raw_results/, processed_results/, etc.).
                                   Only applied if folder_path was not already set by build().

        :param layout_connection:  str — hardware qubit connectivity. One of:
                                       "all_to_all"             (default, fully connected)
                                       "linear_nearest_neighbor" (1D chain, LNN transpilation)
                                   Only applied if lnn_architecture was not already set by build().
                                   Determines which shared circuit folder is used for idempotency checks.

        :param lang:               str — circuit language / execution framework. One of:
                                       "qiskit"  (default)
                                       "myqlm"

        :param service:            str — execution service. One of:
                                       "local"          (default — local simulator)
                                       "IBM_Cloud"            (IBM Quantum, asynchronous)
                                       "MQSS_LRZ"        (IQM EQE1 via MQSS, asynchronous)
                                       "IQM_Resonance"  (IQM Resonance, asynchronous)
                                       "QAAS_IT4I"        (IT4I VLQ via LEXIS/qaas, asynchronous;
                                                         chunks_per_qpu_job_batch chunks per job,
                                                         default VLQ_CHUNKS_PER_JOB)
                                       "PCSS"    (AQT PIAST-Q, synchronous — needs chunk_index or chunks=[nc])

        :param backend:            Backend object for the target QPU. Examples:
                                       AerSamplerV2()                     for service="local"
                                       QiskitRuntimeService(...).backend() for service="IBM_Cloud"
                                   Must not be None.

        :param chunk_index:        int — for the synchronous real-QPU services
                                   (currently AQT_PIAST-Q only) either this or chunks=[nc]
                                   is required. Submits exactly this one
                                   chunk (0-based) and returns; the caller is expected to
                                   invoke submit() once per chunk_index in an external loop
                                   (see source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/qbcs_real_submit_aqt_loop.py),
                                   which is what makes an overnight run resumable after a
                                   crash/reconnection. Unused by local/IBM/IQM_* services.

        :param physical_qubits:    list[int] | None — physical qubits for virtual qubits 0..n-1.
                                   None (default) picks them from the backend's calibration data
                                   (source/core/qpu/layout_selection.py). Either way, every chunk and the SPAM
                                   batch are transpiled onto the same qubits, recorded in
                                   <run>/transpiled_circuits/physical_qubits.json; a resumed run
                                   reuses that record.

        :param max_circuits_per_job: int | None — IQM_EQE / IQM_RESONANCE only. The chunks are
                                   packed into jobs of at most this many circuits, as many whole
                                   chunks (each with its SPAM circuits) per job as fit. None uses
                                   IQM_MAX_CIRCUITS_PER_JOB (Resonance 100, EQE1 70).

        :param chunks:             list[int] | None — the chunks to submit, normally
                                   chunks_to_submit(run_folder, max_chunks), the same list the
                                   submit script gives its health checks. May have gaps (e.g. the
                                   chunks of failed jobs). Chunks already submitted or complete are
                                   still skipped. AQT_PIAST-Q: one chunk ([nc], same as chunk_index=nc).
                                   None: every chunk left (up to max_chunks).
        """
        assert backend is not None, f"backend cannot be {backend}!!"
        if service == "local":
            assert parse_run_name(run_folder).is_simulation, (
                f"run_folder={run_folder!r} must be a simulation's run name (data type sim-…, "
                f"source/core/records/run_names.py) for a local-simulator submission — run_post_process.py and "
                f"the post-processing scripts tell local runs from real ones by it."
            )
        self.CONFIG["service"] = service   # kept in CONFIG only; nothing in the pipeline or analysis reads it back
        # Set run folder
        if "folder_path" not in self.CONFIG or self.CONFIG["folder_path"] is None:
            self.CONFIG["folder_path"] = os.path.join(PROJECT_ROOT, "datafiles", run_folder)

        self.TIMESTAMP_FILE = Path(self.CONFIG["folder_path"]) / TIMESTAMP_FILENAME
        timestamp(label="start", timestamp_file=self.TIMESTAMP_FILE)
        # Set architecture only if not already set by build()
        if "lnn_architecture" not in self.CONFIG or self.CONFIG["lnn_architecture"] is None:
            if layout_connection == "linear_nearest_neighbor":
                self.CONFIG["lnn_architecture"] = True 
            elif layout_connection == "all_to_all":
                self.CONFIG["lnn_architecture"] = False

        assert self._status == "built", (
            f"submit() called with status '{self._status}' — call build() first."
        )

        # What this run is (shots, samples, depths, topology, SPAM) — read back by the analysis scripts (source/core/records/run_info.py)
        # Checked on every submit() call (the same object may submit to several run folders); write_run_info()
        # only touches the file when it is missing or its contents differ, so repeated per-chunk calls
        # don't hammer an unchanged file (which raced with file locks on Windows).
        write_run_info(self.CONFIG, self.CONFIG["folder_path"], self.CONFIG["lnn_architecture"],
                       circuits_qiskit_v13_qpy=self._qiskit_v13_qpy,
                       instance=self.run_instance[0], instance_plan=self.run_instance[1])
        assert hasattr(self, "chunk_paths"), (
            "interleave_job_batches() must be called before submit()."
        )
        assert hasattr(self, "n_chunks"), (
            "compute_chunk_arithmetic() must be called before submit()."
        )
        assert hasattr(self, "chunk_size"), (
           "chunk_size must be assigned before submit()."
        )
        if chunks is not None:
            chunks = sorted(set(chunks))
            assert all(0 <= nc < self.n_chunks for nc in chunks), (
                f"chunks {chunks} out of range — only {self.n_chunks} chunks exist."
            )

        if   lang == "qiskit" and service == "local":
            # ── LOCAL SIMULATOR (synchronous, in-process, called once per run) ─
            self._physical_qubits = None
            if layout_backend is not None:   # only then are circuits transpiled to a device
                self.resolve_physical_qubits(layout_backend, physical_qubits)
            if self.CONFIG["SPAM_correction"]:
                self.build_spam_batch_qiskit()
            _submit_chunk = lambda nc: self.submit_chunk_local_qiskit(
                nc, qiskit_sampler=backend, backend=layout_backend,
            )

            timestamp(label="Starting chunk submission.", timestamp_file=self.TIMESTAMP_FILE)
            for nc in (range(self.n_chunks) if chunks is None else chunks):
                timestamp(label=f"qpu_chunk{nc}_start", timestamp_file=self.TIMESTAMP_FILE)
                _submit_chunk(nc)   # writes job_records/job_record_chunk_<nc>.json, as a real QPU's submission does
                timestamp(label=f"qpu_chunk{nc}_finished",  timestamp_file=self.TIMESTAMP_FILE)
            timestamp(label="All chunks submitted.", timestamp_file=self.TIMESTAMP_FILE)

        elif lang == "qiskit" and service == IBM_CLOUD:
            # ── ASYNCHRONOUS (submits and returns; collect_results() later) ─────
            from source.core.qpu.providers import ibm_batch, connect_ibm, ibm_instance_for_runs
            from source.core.qpu.tokens import read_token

            real_backend = backend
            ibm_service  = None
            if "fake" not in type(backend).__module__.lower() and isinstance(backend, str):
                # a backend given by name: connect to the run's recorded instance, else the Open Plan one
                ibm_service, _ = connect_ibm(read_token("ibm_token"), ibm_instance_for_runs(self.run_instance[0], [run_folder]), backend)
                real_backend   = ibm_service.backend(backend)

            self.resolve_physical_qubits(real_backend, physical_qubits)
            if self.CONFIG["SPAM_correction"]:
                self.transpile_spam_for_backend_to_isa(real_backend)

            timestamp(label="Starting chunk submission.", timestamp_file=self.TIMESTAMP_FILE)
            n_to_submit = self.n_chunks if max_chunks is None else min(max_chunks, self.n_chunks)
            ibm_chunks  = list(range(n_to_submit)) if chunks is None else chunks
            if chunks_per_qpu_job_batch is None or chunks_per_qpu_job_batch==1:
                for nc in ibm_chunks:
                    with ibm_batch(real_backend, max_time=1800) as ibm_sampler:
                        timestamp(label=f"qpu_chunk{nc}_start", timestamp_file=self.TIMESTAMP_FILE)
                        self.submit_chunk_ibm(nc, ibm_sampler, real_backend, service=ibm_service)
                        timestamp(label=f"qpu_chunk{nc}_finished",  timestamp_file=self.TIMESTAMP_FILE)
            else:
                for batch_start_idx in range(0, len(ibm_chunks), chunks_per_qpu_job_batch):
                    chunk_ids   = ibm_chunks[batch_start_idx:batch_start_idx + chunks_per_qpu_job_batch]
                    timestamp(label=f"chunk{chunk_ids[0]} to chunk{chunk_ids[-1]}_start", timestamp_file=self.TIMESTAMP_FILE)
                    with ibm_batch(real_backend) as ibm_sampler:
                        self.submit_batch_ibm(chunk_ids, ibm_sampler, real_backend, service=ibm_service)
                    timestamp(label=f"chunk{chunk_ids[0]} to chunk{chunk_ids[-1]}_end", timestamp_file=self.TIMESTAMP_FILE)
                    gc.collect()
                    print(f"  [IBM] Memory freed after batch {chunk_ids[0]}–{chunk_ids[-1]}")

            timestamp(label="All chunks submitted.", timestamp_file=self.TIMESTAMP_FILE)

        elif lang == "qiskit" and service in (MQSS_LRZ, IQM_RESONANCE, QAAS_IT4I):
            # ── ASYNCHRONOUS (submits and returns; collect_results() later) ────
            # As many whole chunks (each with its SPAM circuits) as fit into one job of
            # at most max_circuits_per_job circuits go into one job. VLQ has no circuit
            # limit: its jobs hold chunks_per_qpu_job_batch chunks (VLQ_CHUNKS_PER_JOB).
            if service == QAAS_IT4I:
                max_circuits_per_job = (chunks_per_qpu_job_batch or VLQ_CHUNKS_PER_JOB) * self._circuits_per_chunk()
            elif max_circuits_per_job is None:
                max_circuits_per_job = IQM_MAX_CIRCUITS_PER_JOB[service]
            self.resolve_physical_qubits(backend, physical_qubits)
            if self.CONFIG["SPAM_correction"]:
                self.transpile_spam_for_backend_to_isa(backend)

            timestamp(label="Starting chunk submission.", timestamp_file=self.TIMESTAMP_FILE)
            n_to_submit = self.n_chunks if max_chunks is None else min(max_chunks, self.n_chunks)
            job_records = Path(get_run_subfolder(self.CONFIG, "job_records"))

            def _submitted(nc):
                rp = job_records / f"job_record_chunk_{nc}.json"
                return rp.exists() and json.loads(rp.read_text(encoding="utf-8"))["status"] in ("submitted", "complete")

            todo      = [nc for nc in (range(n_to_submit) if chunks is None else chunks)
                         if not _submitted(nc)]   # a resumed run packs only what is left
            n_per_job = self._chunks_per_iqm_job(max_circuits_per_job)
            print(f"  [{service}] {len(todo)} chunk(s) to submit, {n_per_job} per job "
                  f"(≤ {max_circuits_per_job} circuits per job)")
            for i in range(0, len(todo), n_per_job):
                chunk_ids = todo[i:i + n_per_job]
                timestamp(label=f"chunk{chunk_ids[0]} to chunk{chunk_ids[-1]}_start", timestamp_file=self.TIMESTAMP_FILE)
                self.submit_batch_iqm(chunk_ids, backend, service, max_circuits_per_job)
                timestamp(label=f"chunk{chunk_ids[0]} to chunk{chunk_ids[-1]}_end", timestamp_file=self.TIMESTAMP_FILE)

            timestamp(label="All chunks submitted.", timestamp_file=self.TIMESTAMP_FILE)
            print(f"  [{service}] All jobs submitted.")


        elif lang == "qiskit" and service == PCSS:
            # ── SYNCHRONOUS REAL QPU ──────────────────────────────────────────
            # submit_chunk_aqt() blocks until the chunk's results are back, so this
            # branch submits exactly one chunk per submit() call (chunk_index) rather
            # than looping — the caller (qbcs_real_submit_aqt_loop.py) drives the
            # per-chunk loop externally, one submit() call per chunk, so a crash mid
            # -run only loses the in-flight chunk.
            if chunks is not None:
                assert len(chunks) == 1 and chunk_index in (None, chunks[0]), (
                    f"AQT_PIAST-Q submits one chunk per submit() call — got chunks={chunks}, chunk_index={chunk_index}."
                )
                chunk_index = chunks[0]
            assert chunk_index is not None, (
                "AQT_PIAST-Q is a synchronous-QPU service — submit() requires chunks=[nc] (or chunk_index)."
            )
            nc = chunk_index
            assert 0 <= nc < self.n_chunks, (
                f"chunk_index {nc} out of range — only {self.n_chunks} chunks exist."
            )

            # ── Physical qubits + SPAM transpilation (both idempotent) ────────
            self.resolve_physical_qubits(backend, physical_qubits)
            if self.CONFIG["SPAM_correction"]:
                self.transpile_spam_for_backend_to_isa(backend)

            # ── Bookend "Starting chunk submission." — once per process ───────
            if not getattr(self, "_sync_submission_started", False):
                timestamp(label="Starting chunk submission.", timestamp_file=self.TIMESTAMP_FILE)
                self._sync_submission_started = True

            # ── Submit this one chunk — submit_chunk_aqt() is idempotent and ───
            # ── prints its own skip message if already done; still logged. ────
            timestamp(label=f"qpu_chunk{nc}_start", timestamp_file=self.TIMESTAMP_FILE)
            self.submit_chunk_aqt(nc, backend)
            timestamp(label=f"qpu_chunk{nc}_finished", timestamp_file=self.TIMESTAMP_FILE)

            # ── Bookend "All chunks submitted." — once, on the real last chunk ─
            if nc == self.n_chunks - 1:
                timestamp(label="All chunks submitted.", timestamp_file=self.TIMESTAMP_FILE)
                print(f"  [{PCSS}] All chunks submitted.")

            gc.collect()

        else:
            raise NotImplementedError(f"lang={lang}, service={service} not yet implemented.")


        if service == "local":
            # -------- Merge all chunks --------------------------------
            self.merge_all_chunks()

            # ── Build SPAM confusion matrix after all chunks ──────────────────────
            if self.CONFIG["SPAM_correction"]:
                timestamp(label="Computing SPAM confusion matrix.", timestamp_file=self.TIMESTAMP_FILE)
                compute_and_save_spam_confusion_matrix(config=self.CONFIG)

        timestamp(label="end", timestamp_file=self.TIMESTAMP_FILE)
        self._status = "submitted"

        return Path(self.CONFIG["folder_path"])

    def _spam_qpy_path(self, kind=None):
        """Shared-cache path of the raw SPAM circuits: spam_qiskit.qpy next to the
        chunk_{nc}_qiskit.qpy files, or spam_qiskit_v13.qpy (QPY version 13) next to the
        chunk_{nc}_qiskit_v13.qpy ones. kind defaults to the build's format."""
        if kind is None:
            kind = "qiskit_v13" if self._qiskit_v13_qpy else "qiskit"
        circ_top = get_circuit_folder(self.CONFIG, top_folder=True)
        return Path(get_chunk_folder(circ_top, kind)) / f"spam_{kind}.qpy"

    def _convert_spam_myqlm_to_qiskit(self):
        """The 2^nsites SPAM calibration circuits, built in myQLM and converted to
        Qiskit (with measure_all). The only place SPAM circuits are made."""
        require_myqlm("Building the SPAM calibration circuits (spam_*.qpy is missing)")
        from qat.qiskit.converters import QaptivaToQiskit
        spam_batch = build_spam_calibration_batch(
            nqubits         = self.CONFIG["nsites"],
            shots_per_state = self.CONFIG["SPAM_circuit_shots_per_input_state"]
        )
        translator  = QaptivaToQiskit()
        qk_circuits = [translator.to_circ(job.circuit) for job in spam_batch.jobs]
        for qk_circ in qk_circuits:
            qk_circ.measure_all()
        return qk_circuits

    def convert_spam_to_qiskit(self):
        """Build stage: save the raw SPAM circuits to the shared circuit cache
        (_spam_qpy_path()), in the same format as convert_chunks_to_qiskit()'s
        chunks, so submitting never has to build them in myQLM. Skipped when
        SPAM_correction is off or the file already exists."""
        if not self.CONFIG["SPAM_correction"]:
            return
        qpy_path = self._spam_qpy_path()
        if qpy_path.exists():
            return

        from qiskit.qpy import dump as qpy_dump
        qk_circuits = self._convert_spam_myqlm_to_qiskit()
        qpy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qpy_path, "wb") as f:
            if self._qiskit_v13_qpy:
                qpy_dump(qk_circuits, f, version=13)
            else:
                qpy_dump(qk_circuits, f)
        print(f"  [SPAM] Saved {len(qk_circuits)} calibration circuits → {rel_path(qpy_path)}")

    def build_spam_batch_qiskit(self, kind=None):
        """
        Load the raw SPAM calibration circuits (with measurements) into
        self._qk_spam_circuits — from the shared cache written by build()
        (convert_spam_to_qiskit()), or, for a cache built before that stage
        existed, by converting them from myQLM here.
        """
        assert self.CONFIG["SPAM_correction"], (
            "SPAM_correction is False — no need to call build_spam_batch_qiskit()."
        )
        qpy_path = self._spam_qpy_path(kind)
        if qpy_path.exists():
            from qiskit.qpy import load
            with open(qpy_path, "rb") as f:
                self._qk_spam_circuits = load(f)
            print(f"  [SPAM] Loaded {len(self._qk_spam_circuits)} calibration circuits from {qpy_path.name}.")
        else:
            print(f"  ⚠️  [SPAM] {rel_path(qpy_path)} not found — converting from myQLM instead. "
                  f"Run build() in local_sim.venv to cache it.")
            self._qk_spam_circuits = self._convert_spam_myqlm_to_qiskit()
            print(f"  [SPAM] Converted {len(self._qk_spam_circuits)} calibration circuits to Qiskit.")

    def merge_all_chunks(self):
        """
        Merges per-chunk JSON files into merged JSON files for analysis.
        Wraps merge_chunks() from source/core/analysis/mitigation.py.
        Always rewrites the merged files.
        """
        merged_counts_path   = get_run_subfolder(self.CONFIG, "raw_results", "results_counts_merged.json")
        merged_fidelity_path = get_run_subfolder(self.CONFIG, "raw_results", "results_fidelity_merged.json")

        # ── Load per-chunk JSONs ──────────────────────────────────────────────
        timestamp(label="Merging chunks.", timestamp_file=self.TIMESTAMP_FILE)

        measurement_counts = load_json(get_run_subfolder(self.CONFIG, "raw_results", "results_counts.json"))
        fidelity_nec_dict  = load_json(get_run_subfolder(self.CONFIG, "raw_results", "results_fidelity.json"))

        # ── Merge ─────────────────────────────────────────────────────────────
        measurement_counts = merge_chunks(measurement_counts)
        fidelity_nec_dict  = merge_chunks(fidelity_nec_dict)

        # ── Save merged files ─────────────────────────────────────────────────
        with open(merged_counts_path, "w", encoding="utf-8") as f:
            json.dump({"chunk_0": measurement_counts}, f, indent=4)
        print(f"  [merge_all_chunks] Saved merged counts → {rel_path(merged_counts_path)}")

        with open(merged_fidelity_path, "w", encoding="utf-8") as f:
            json.dump({"chunk_0": fidelity_nec_dict}, f, indent=4)
        print(f"  [merge_all_chunks] Saved merged fidelity → {rel_path(merged_fidelity_path)}")

        # ── Copy to local working directory ───────────────────────────────────
        # shutil.copy2(merged_counts_path,   "./results_counts_merged.json")
        # shutil.copy2(merged_fidelity_path, "./results_fidelity_merged.json")
        # print("  [merge_all_chunks] Local copies saved to ./")

        timestamp(label="Merging done.", timestamp_file=self.TIMESTAMP_FILE)


    def analyze_results(self, n_samples=None, spam_n_samples=None):
        """
        Computes exact observables, mitigated expectation values, AWAE and saves results.
        Requires merge_all_chunks() to have been called first.

        n_samples: analyze only the first n_samples chunks instead of the full
        self.CONFIG["samples"] (None = the full run; an incomplete run is cut to the
        chunks collected so far, with a warning).
        Reads the same raw_results/*_merged.json — nothing is copied or
        re-merged. Writes to "+nsamples-{n_samples}"-suffixed filenames in
        processed_results/ alongside the full run's own outputs — safe
        to call repeatedly with different n_samples on an already-analysed run.
        A subset varies only the RC samples: the SPAM correction uses every
        collected chunk's calibration (the run's own SPAM confusion matrix)
        unless spam_n_samples (below the collected chunks) asks for fewer, which
        also writes a "+nsamples-{spam_n_samples}" SPAM matrix.
        """
        assert self._status == "submitted", (
            f"analyze_results() called with status '{self._status}' — call submit() first."
        )
        if n_samples is None:
            # An incomplete run (jobs still running or failed): analyze the chunks collected so far,
            # into the "+nsamples-{collected}" files; what is missing is listed in their completeness block.
            from source.core.records.chunk_scope import collected_chunks
            n_done = len(collected_chunks(self.CONFIG["folder_path"]))
            if 0 < n_done < self.CONFIG["samples"]:
                print(f"  ⚠️  Only {n_done} of {self.CONFIG['samples']} chunks are collected — analyzing those.")
                n_samples = n_done
        assert n_samples is None or 0 < n_samples <= self.CONFIG["samples"], (
            f"n_samples must be in (0, {self.CONFIG['samples']}], got {n_samples}"
        )
        # ── Precondition check ────────────────────────────────────────────────
        merged_counts_path   = get_run_subfolder(self.CONFIG, "raw_results", "results_counts_merged.json")
        merged_fidelity_path = get_run_subfolder(self.CONFIG, "raw_results", "results_fidelity_merged.json")        

        assert os.path.exists(merged_counts_path),   "results_counts_merged.json missing — call merge_all_chunks() first."
        assert os.path.exists(merged_fidelity_path), "results_fidelity_merged.json missing — call merge_all_chunks() first."

        check_shots_consistency(self.CONFIG)

        timestamp(label="Starting analysis.", timestamp_file=self.TIMESTAMP_FILE)

        # ── Step 5: Save ──────────────────────────────────────────────────────
        use_circuit_LNN = self._use_circuit_LNN

        if self.CONFIG.get("SPAM_correction", False):
            # With SPAM correction — QPU/total runtime is identical either way
            # (QPU time comes from the job records, CPU time from the timestamp log,
            # neither from SPAM_correction), so skip the printout here and let the
            # non-SPAM call below print it once.
            post_processing_and_saving(CONFIG=self.CONFIG, roles=self.roles,
                                       use_circuit_LNN=use_circuit_LNN,
                                       filename="run_data_spam.json", n_samples=n_samples,
                                       spam_n_samples=spam_n_samples,
                                       print_runtimes=False, qiskit_v13_qpy=self._qiskit_v13_qpy)
            # Without SPAM correction
            self.CONFIG["SPAM_correction"] = False
            post_processing_and_saving(CONFIG=self.CONFIG, roles=self.roles,
                                       use_circuit_LNN=use_circuit_LNN,
                                       filename="run_data.json", n_samples=n_samples,
                                       qiskit_v13_qpy=self._qiskit_v13_qpy)
            self.CONFIG["SPAM_correction"] = True
        else:
            post_processing_and_saving(CONFIG=self.CONFIG, roles=self.roles,
                                       use_circuit_LNN=use_circuit_LNN, n_samples=n_samples,
                                       qiskit_v13_qpy=self._qiskit_v13_qpy)

        timestamp(label="End of Analysis", timestamp_file=self.TIMESTAMP_FILE)
        self._status = "analysed"

    def build(self, circuit_transpilation_LNN=False):
        """
        The chunk format follows the topology. LNN circuits are saved as
        chunks_qiskit/chunk_{nc}_qiskit.qpy at Qiskit's default QPY version.
        all_to_all circuits are saved as chunks_qiskit_v13_qpy/chunk_{nc}_qiskit_v13.qpy
        at QPY version 13, which the older Qiskit of aqt.venv (1.4.6) and
        iqm_vlq.venv (2.1) can read; every newer Qiskit reads it too. The raw SPAM
        circuits are saved beside the chunks in the same format (spam_qiskit_v13.qpy).
        Timestamps of an all_to_all build go to timestamps_qiskit_v13_qpy_local_time.csv
        (and its _utc companion).
        """
        assert self._status == "initialised", (
            f"build() called on a pipeline with status '{self._status}'. Reinitialise to rebuild."
        )

        self._qiskit_v13_qpy            = not circuit_transpilation_LNN
        self.CONFIG["lnn_architecture"] = circuit_transpilation_LNN
        self.compute_chunk_arithmetic()
        if os.environ.get(CIRCUITS_ROOT_ENV):   # only the test suite sets it: make a leaked value visible
            print(f"  [build] Circuit caches in {circuits_root()} ({CIRCUITS_ROOT_ENV})")
        check_circuit_cache_config(self.CONFIG)   # refuses a cache built with other samples/rng_seed/dt/code version

        missing = self._missing_qiskit_cache_files(circuit_transpilation_LNN)
        if not missing:
            self._build_from_qiskit_cache(circuit_transpilation_LNN)
        else:
            require_myqlm(f"build() ({len(missing)} circuit-cache file(s) to create, e.g. {missing[0]})")
            self.build_qbcs_circuits(circuit_transpilation_LNN=circuit_transpilation_LNN, rotate_qubits_locally=True)
            self.build_all_nec_ems_job_batches()
            self.interleave_job_batches()
            self.convert_chunks_to_qiskit()
            self.convert_spam_to_qiskit()
            cache_exact_observables(self.CONFIG, self.trotter_steps_list, self._use_circuit_LNN)
        self._cleanup_empty_circuit_folders()
        self._status = "built"

    def _missing_qiskit_cache_files(self, circuit_transpilation_LNN):
        """What a submission, collection or analysis needs from the circuit cache
        (this build format's chunk QPY files, the SPAM circuits and the exact
        observables) that isn't there yet. Empty when build() can skip myQLM."""
        circ_top = get_circuit_folder(self.CONFIG, top_folder=True)
        migrate_flat_chunk_files(circ_top)
        kind     = "qiskit_v13" if self._qiskit_v13_qpy else "qiskit"
        folder   = Path(get_chunk_folder(circ_top, kind))
        missing  = [str(p) for p in (folder / f"chunk_{nc}_{kind}.qpy" for nc in range(self.n_chunks)) if not p.exists()]
        if self.CONFIG["SPAM_correction"] and not self._spam_qpy_path().exists():
            missing.append(str(self._spam_qpy_path()))
        nts = missing_exact_observables(self.CONFIG, self.trotter_steps_list, circuit_transpilation_LNN)
        if nts:
            missing.append(f"{exact_observables_path(self.CONFIG)} (Nt={nts})")
        return missing

    def _build_from_qiskit_cache(self, circuit_transpilation_LNN):
        """build() when every file a run needs is already cached as Qiskit QPY/JSON:
        sets up the same state as the full build without loading myQLM, so the
        real-QPU venvs never import it."""
        circuit_folder  = get_circuit_folder(self.CONFIG, top_folder=True)
        target_filename = TIMESTAMP_FILENAME_V13 if self._qiskit_v13_qpy else TIMESTAMP_FILENAME
        sync_circuit_timestamp_logs(circuit_folder, self._qiskit_v13_qpy)
        self.TIMESTAMP_FILE    = Path(circuit_folder) / target_filename
        self._circuits_rotated = True
        self._use_circuit_LNN  = circuit_transpilation_LNN
        self._reset_config_depth()
        myqlm_chunk_folder = get_chunk_folder(circuit_folder, "myqlm")
        self.chunk_paths   = [os.path.join(myqlm_chunk_folder, f"batch_jobs_interleaved_chunk{nc}")
                              for nc in range(self.n_chunks)]   # only for the myQLM fallback

        # Same guard as build_qbcs_circuits()' cached-batch check: a cache built
        # for a different number of depths has chunks of a different size.
        from qiskit.qpy import load
        kind = "qiskit_v13" if self._qiskit_v13_qpy else "qiskit"
        with open(Path(get_chunk_folder(circuit_folder, kind)) / f"chunk_0_{kind}.qpy", "rb") as f:
            n_circuits = len(load(f))
        assert n_circuits == self.chunk_size, (
            f"Cached chunk_0_{kind}.qpy in {circuit_folder} has {n_circuits} circuits, but the current "
            f"config expects {self.chunk_size} ({self.n_depths} depths) — stale/mismatched cache."
        )
        print(f"  [build] Every circuit is cached as Qiskit QPY in {rel_path(circuit_folder)} — myQLM not needed.")

    def _reset_config_depth(self):
        self.CONFIG["ind_T"] = None
        self.CONFIG["T"]     = None
        self.CONFIG["Nt"]    = None

    def get_results_by_chunk_ibm(self, exp_results_folder_path, nc, service=None, poll_interval=30, lang=None, backend_name=None):
        """
        Collect results for chunk nc from IBM backend and save to disk.

        :param exp_results_folder_path: str — run folder path (returned by submit())
        :param nc:                      int — chunk index
        :param service:                 QiskitRuntimeService — required for real IBM, None for fake
        :param poll_interval:           int — seconds between status polls (real IBM only)
        :param lang: str — "qiskit" (bitstrings reversed k[::-1]) | "myqlm" (no reversal)
        """
        assert lang in ("qiskit", "myqlm"), f"lang must be 'qiskit' or 'myqlm', got {lang!r}"

        exp_results_folder_path = Path(exp_results_folder_path)   # ensure Path object
        record_path        = exp_results_folder_path / "job_records" / f"job_record_chunk_{nc}.json"
        record             = json.loads(record_path.read_text(encoding="utf-8"))
        job_id             = record["job_id"]
        n_spam             = record["n_spam"]
        n_circuits         = record["n_circuits"]
        trotter_steps_list = record["trotter_steps_list"]
        roles              = record["roles"]
        nsites             = record["nsites"]
        spam_correction    = record["n_spam"] > 0 
        spam_start_idx     = record.get("spam_start", 0)
        circ_start_idx     = record.get("circ_start", n_spam)
        circ_end_idx       = record.get("circ_end",   n_spam + n_circuits)
        chunk_key          = f"chunk_{nc}"  

        assert abs((circ_start_idx-spam_start_idx)-n_spam) <1e-8
        assert abs((circ_end_idx-circ_start_idx)-n_circuits) <1e-8

        raw_dir       = exp_results_folder_path / "raw_results"
        counts_file   = raw_dir / "results_counts.json"
        fidelity_file = raw_dir / "results_fidelity.json"

        if record["status"] == "complete" and _chunk_in(counts_file, f"chunk_{nc}") and _chunk_in(fidelity_file, f"chunk_{nc}"):
            print(f"  [Chunk {nc}] Already collected — skipping."); return

        if service is not None:
            # Real IBM — reconstruct from cloud via job_id
            job = service.job(job_id)
            # Real IBM — poll until done
            printed = self.__dict__.setdefault("_job_lines_printed", set())      # a batch job holds several chunks: say it once
            if job_id not in printed:
                printed.add(job_id)
                print_job_line(f"Chunk {nc}", backend_name or record["backend_name"], record.get("service_name", IBM_CLOUD), job_id)
            while True:
                status_str = str(job.status()).upper()
                print(f"  [Chunk {nc}] Status: {status_str}")
                state = job_state_of_status(status_str)
                if state == "done":
                    break
                if state == "failed":
                    raise RuntimeError(f"Job {job_id} failed with status: {status_str}")
                time.sleep(poll_interval)
        else:
            # Fake backend — retrieve from local registry
            job = self._fake_job_registry.get(job_id)
            if job is None:
                raise RuntimeError(
                    f"Job {job_id} not found in local registry. "
                    f"Did submit_chunk_ibm() run in this session?"
                ) 
            # Fake backend — synchronous, no polling needed
            print(f"  [Chunk {nc}] Fake backend — collecting immediately.")

        result     = job.result()

        rdict = result[spam_start_idx].data
        if hasattr(rdict, 'meas'):
            spam_raw   = [result[spam_start_idx + si].data.meas.get_counts() for si in range(n_spam)]
            all_counts = [result[circ_start_idx + ci].data.meas.get_counts() for ci in range(n_circuits)]
        elif hasattr(rdict, 'c'):
            spam_raw   = [result[spam_start_idx + si].data.c.get_counts() for si in range(n_spam)]
            all_counts = [result[circ_start_idx + ci].data.c.get_counts() for ci in range(n_circuits)]
        else:
            raise RuntimeError(f"Unknown classical register in result. Available: {list(rdict.__dict__.keys())}")

        measurement_counts = {}
        fidelity_nec_dict  = {}

        for di, nt in enumerate(trotter_steps_list):
            tag = f"Nt_{nt:02d}"
            for ri, role in enumerate(roles):
                # local_idx: ranges 0 to n_depths * len(self.roles) - 1
                # identical across all chunks — same keys appear in every chunk_{nc}
                local_idx = di * len(roles) + ri
                job_key   = f"job_{tag}_{role}_ems{local_idx}"
                circ_key  = f"{tag}_{role}_circuit_0"   # always 0 — sample index inside a chunk (1 sample per chunk)
                if lang=="qiskit": 
                    counts    = {k[::-1]: v for k, v in all_counts[local_idx].items()}
                elif lang=="myqlm":  
                    counts    = {k: v for k, v in all_counts[local_idx].items()}

                if role.endswith("_nec"):
                    fidelity_nec_dict[job_key]  = {circ_key: counts}
                else:
                    measurement_counts[job_key] = {circ_key: counts}

        (exp_results_folder_path / "raw_results").mkdir(parents=True, exist_ok=True)

        counts_path   = exp_results_folder_path / "raw_results" / "results_counts.json"
        fidelity_path = exp_results_folder_path / "raw_results" / "results_fidelity.json"

        existing_counts   = load_json(counts_path)   if counts_path.exists()   else {}
        existing_fidelity = load_json(fidelity_path) if fidelity_path.exists() else {}

        existing_counts[chunk_key]   = measurement_counts
        existing_fidelity[chunk_key] = fidelity_nec_dict

        with open(counts_path, "w", encoding="utf-8") as f:
            json.dump(existing_counts, f, indent=4)
        with open(fidelity_path, "w", encoding="utf-8") as f:
            json.dump(existing_fidelity, f, indent=4)

        if spam_correction:
            spam_path = exp_results_folder_path / "raw_results" / "spam_calibration_counts_per_chunk.json"
            existing_spam = load_json(spam_path) if spam_path.exists() else {}
            if lang=="qiskit":
                spam_counts_per_input = {
                    format(si, f"0{nsites}b"): {k[::-1]: v for k, v in spam_raw[si].items()}
                    for si in range(n_spam)
                }
            elif lang=="myqlm":
                spam_counts_per_input = {
                    format(si, f"0{nsites}b"): {k: v for k, v in spam_raw[si].items()}
                    for si in range(n_spam)
                }                
            existing_spam[chunk_key] = spam_counts_per_input
            with open(spam_path, "w", encoding="utf-8") as f:
                json.dump(existing_spam, f, indent=4)

        record["status"] = "complete"

        created_at = scheduled_at = finished_at = "N/A"
        if service is not None:
            t = ibm_collect_timing(job, result)
            merge_timing(record, **t)
            created_at   = t.get("created_at") or "N/A"
            scheduled_at = t.get("execution_start") or t.get("running_at") or "N/A"   # QPU execution start
            finished_at  = t.get("finished_at") or "N/A"
            check = self._check_executed_ibm(job, record, exp_results_folder_path, nc)
        else:
            check = CC.not_available("simulated backend (fake/Aer): there is no server copy of the circuits")
        record["circuit_check"] = check
        _warn_circuit_check(f"Chunk {nc}", check)

        record["created_at"]      = created_at
        record["scheduled_at"]    = scheduled_at
        record["finished_at"]     = finished_at
        record["result_path"]     = run_relative_path(exp_results_folder_path, "raw_results", "results_counts.json")
        write_job_record(record_path, record)
        
        print_chunk_saved(nc, n_spam if spam_correction else 0, len(measurement_counts), len(fidelity_nec_dict))

    def _submitted_circuits(self, run_path, record, nc):
        """The ISA circuits a chunk job was built from (SPAM first), read back from
        transpiled_circuits/ — exactly what submit_chunk_*() sent."""
        n_spam = record.get("n_spam", 0)
        files  = record.get("submitted_circuit_files") or self._submitted_circuit_files(nc, n_spam)
        circuits = []
        for fname in files:
            circuits += CC.load_qpy(Path(run_path) / "transpiled_circuits" / fname)
        return circuits

    def _check_executed_ibm(self, job, record, run_path, nc):
        """Compare job.inputs['pubs'] (the circuits IBM ran) with the submitted ones. Never raises."""
        try:
            cache = self.__dict__.setdefault("_ibm_inputs_cache", {})
            if record["job_id"] not in cache:          # a batched job is shared by several chunks
                cache.clear()
                cache[record["job_id"]] = job.inputs["pubs"]
            pubs  = cache[record["job_id"]]
            start = record.get("spam_start", 0)
            end   = record.get("circ_end", start + record["n_spam"] + record["n_circuits"])
            return CC.check_ibm(self._submitted_circuits(run_path, record, nc), pubs[start:end], record["nshots"])
        except Exception as e:
            return CC.check_failed("IBM job.inputs['pubs'] vs submitted ISA circuits", e)

    def _check_executed_iqm_resonance(self, backend, record, run_path, nc):
        """Compare the IQM Resonance job payload (the circuits IQM received) with the submitted ones.
        Never raises."""
        try:
            cache = self.__dict__.setdefault("_iqm_payload_cache", {})
            if record["job_id"] not in cache:          # a packed job is shared by several chunks
                cache.clear()
                cache[record["job_id"]] = iqm_raw_job(backend, record["job_id"], "/payload")
            payload = cache[record["job_id"]]
            start   = record.get("spam_start", 0)
            end     = record.get("circ_end", start + record["n_spam"] + record["n_circuits"])
            return CC.check_iqm(self._submitted_circuits(run_path, record, nc), payload,
                                backend.index_to_qubit_name, record["nshots"], circuit_slice=slice(start, end))
        except Exception as e:
            return CC.check_failed("IQM GET jobs/{id}/payload vs submitted ISA circuits", e)

    def _check_executed_iqm_vlq(self, job, record, run_path, nc):
        """Compare the circuits VLQ's server ran (qaas job.get_transpiled_circuits()) with the
        submitted ones. Never raises."""
        method = "VLQ job.get_transpiled_circuits() vs submitted ISA circuits"
        try:
            start = record.get("spam_start", 0)
            end   = record.get("circ_end", start + record["n_spam"] + record["n_circuits"])
            return CC.check_vlq(self._submitted_circuits(run_path, record, nc),
                                VLQ.transpiled_circuits(job, record["job_id"])[start:end], method)
        except Exception as e:
            return CC.check_failed(method, e)

    def collect_results(self, exp_results_folder_path, service=None, lang="qiskit", poll_interval=30, service_name=None,
                        wait=False, backend_name=None):
        """
        Collect the results of every finished chunk job and merge them. Works for IBM and IQM backends.

        Non-blocking: each job's status is queried once; finished jobs are collected in whatever
        order they finished, unfinished ones are left for a later call, failed ones are marked
        failed. Each chunk's collection (result download, circuit check, saving) is logged as
        chunk{nc}_collect_start/_end and stored as timing.collect_s in its record.

        :param exp_results_folder_path: str/Path — run folder (returned by submit())
        :param service:                 QiskitRuntimeService — IBM, MQSSQiskitAdapter — IQM
        :param lang:                    "qiskit" | "myqlm" — bitstring reversal
        :param poll_interval:           int — seconds between passes when wait=True
        :param service_name:            str - "IBM_Cloud" / "MQSS_LRZ" / "IQM_Resonance" / "QAAS_IT4I"
                                        (IQM_RESONANCE / IQM_VLQ: `service` is the backend)
        :param wait:                    True: repeat the passes until no job is pending
        :param backend_name:            str — the script's BACKEND_NAME, for the progress lines (default: the record's)
        :returns:                       list of the chunks still pending
        """
        assert service_name is not None, "service_name cannot be None!"

        exp_results_folder_path = Path(exp_results_folder_path)
        job_records_dir         = exp_results_folder_path / "job_records"

        assert job_records_dir.exists(), \
            f"No job_records/ found in {exp_results_folder_path}"

        from source.core.records.job_records import chunk_record_files
        record_files = list(chunk_record_files(job_records_dir).values())   # sorted by chunk index

        assert len(record_files) > 0, \
            f"No job records found in {job_records_dir}"

        # ── Restore CONFIG from first record ─────────────────────────────────
        first_record = json.loads(record_files[0].read_text(encoding="utf-8"))
        self.CONFIG.update({k: v for k, v in first_record["CONFIG"].items()
                            if k not in MACHINE_SPECIFIC_CONFIG_KEYS})   # old records hold the writer's absolute path

        print(f"  [Collect] Collecting results for {len(record_files)} chunks...")

        raw_dir       = exp_results_folder_path / "raw_results"
        counts_file   = raw_dir / "results_counts.json"
        fidelity_file = raw_dir / "results_fidelity.json"

        if service_name not in (IBM_CLOUD, MQSS_LRZ, IQM_RESONANCE, QAAS_IT4I):
            raise ValueError(f"Unknown service_name: {service_name!r}. Must be 'IBM_Cloud' / 'MQSS_LRZ' / 'IQM_Resonance' / 'QAAS_IT4I'.")
        self.TIMESTAMP_FILE = Path(exp_results_folder_path) / TIMESTAMP_FILENAME   # collection labels go into the run's log
        failed_timing = {}   # job_id -> timing of a failed job, queried once

        while True:
            pending, n_collected, n_skipped, states = [], 0, 0, {}   # states: job_id -> (state, detail), one query per job
            for record_file in record_files:
                record = json.loads(record_file.read_text(encoding="utf-8"))
                nc     = record["chunk_id"]

                if record["status"] == "failed" or (record["status"] == "complete" and _chunk_in(counts_file, f"chunk_{nc}") and _chunk_in(fidelity_file, f"chunk_{nc}")):
                    n_skipped += 1
                    continue

                job_id = record["job_id"]
                if job_id not in states:
                    states[job_id] = _remote_job_state(service_name, service, job_id,
                                                       record.get("backend_name"), record.get("nshots"))
                state, detail = states[job_id]
                if state == "pending":
                    pending.append(nc)
                    continue
                if state == "failed":
                    record["status"], record["failure"] = "failed", f"job ended {detail}"
                    record["failure_stage"] = "execution"
                    if job_id not in failed_timing:   # one query per job (batched chunks share it)
                        failed_timing[job_id] = _failed_job_timing(service_name, service, record)
                    merge_timing(record, **failed_timing[job_id])
                    t = failed_timing[job_id]
                    record["finished_at"] = t.get("finished_at") or t.get("execution_end") or t.get("completed_at") or record.get("finished_at", "N/A")
                    if t.get("execution_start") and record.get("scheduled_at") in (None, "N/A"):
                        record["scheduled_at"] = t["execution_start"]
                    write_job_record(record_file, record)
                    print(f"  ⚠️  [Chunk {nc}] Job {job_id} ended {detail} — marked failed.")
                    continue

                self._log_timestamp(f"chunk{nc}_collect_start")
                t0 = time.perf_counter()
                try:
                    self._collect_chunk(service_name, exp_results_folder_path, nc, service, lang, poll_interval, backend_name)
                except Exception as e:
                    print(f"  ⚠️  [Chunk {nc}] Not collected: {type(e).__name__}: {e}")
                    pending.append(nc)
                    continue
                collect_s = time.perf_counter() - t0
                self._log_timestamp(f"chunk{nc}_collect_end")
                record = json.loads(record_file.read_text(encoding="utf-8"))
                merge_timing(record, collect_s=round(collect_s, 3))
                write_job_record(record_file, record)
                n_collected += 1

            print(f"  [Collect] {n_collected} chunk(s) collected now, {n_skipped} already done or failed, "
                  f"{len(pending)} pending" + (f": {format_chunk_range(pending)}" if pending else "") + ".")
            if not (wait and pending):
                break
            time.sleep(poll_interval)

        if n_collected == 0 and not (Path(exp_results_folder_path) / "raw_results" / "results_counts.json").exists():
            print(f"  [{service_name}] Nothing collected yet — no job has finished.")
            return pending

        # ── Set folder path for merge ─────────────────────────────────────────
        self.CONFIG["folder_path"] = str(exp_results_folder_path)
        # ── Merge all chunks ──────────────────────────────────────────────────
        self.merge_all_chunks()

        # ── SPAM confusion matrix ─────────────────────────────────────────────
        if self.CONFIG["SPAM_correction"]:
            compute_and_save_spam_confusion_matrix(config=self.CONFIG)

        if pending:
            print(f"  [{service_name}] Collected chunks merged; chunks {format_chunk_range(pending)} are still "
                  f"pending — run the collection again later.")
        else:
            print(f"  [{service_name}] All finished chunks collected and merged.")
        return pending

    def _collect_chunk(self, service_name, exp_results_folder_path, nc, service, lang, poll_interval, backend_name=None):
        """Collect one chunk whose job has finished (collect_results())."""
        if service_name == MQSS_LRZ:
            self.get_results_by_chunk_iqm(exp_results_folder_path=exp_results_folder_path, nc=nc,
                                          adapter=service, lang=lang, backend_name=backend_name)
        elif service_name in (IQM_RESONANCE, QAAS_IT4I):
            self.get_results_by_chunk_iqm_resonance(exp_results_folder_path=exp_results_folder_path, nc=nc,
                                                    backend=service, lang=lang, backend_name=backend_name)   # backend: retrieve_job()
        else:   # IBM
            self.get_results_by_chunk_ibm(exp_results_folder_path=exp_results_folder_path, nc=nc,
                                          service=service, lang=lang, poll_interval=poll_interval, backend_name=backend_name)

    def reset_for_resubmit(self):
        """Reset status to allow resubmission with different shots."""
        self._status = "built"
        self.CONFIG["folder_path"] = None
        self.TIMESTAMP_FILE        = None
        self._fake_job_registry    = {}
        print(f"  ⚠️  [Submit] The next submit will take circuits with linear nearest neighbor = {self.CONFIG["lnn_architecture"]}.")


    def submit_batch_ibm(self, batch_chunk_ids, sampler, backend, service=None):
        """
        Transpile N chunks and submit as a single IBM job.
        Each chunk contributes n_spam + n_circuits circuits to the job.
        Saves one job record per chunk, all pointing to the same job_id,
        with spam_start/circ_start/circ_end offsets for result extraction.

        :param batch_chunk_ids: list of int — chunk indices to submit together
        :param sampler:         qiskit_ibm_runtime.SamplerV2
        :param backend:         IBM backend object
        :param service:         QiskitRuntimeService — required for real IBM, None for fake
        """
        from qiskit.qpy import load
        import time

        # ── Idempotency — skip if all chunks already submitted ────────────────
        all_done = True
        for nc in batch_chunk_ids:
            record_path = Path(get_run_subfolder(self.CONFIG, "job_records")) / f"job_record_chunk_{nc}.json"
            if not record_path.exists():
                all_done = False
                break
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["status"] not in ("submitted", "complete"):
                all_done = False
                break
        if all_done:
            print(f"  [Batch {batch_chunk_ids[0]}–{batch_chunk_ids[-1]}] All already submitted — skipping.")
            return

        # ── Load SPAM ISA circuits once ───────────────────────────────────────
        n_spam   = 0
        spam_isa = []
        if self.CONFIG["SPAM_correction"]:
            spam_qpy = Path(get_run_subfolder(self.CONFIG, "transpiled_circuits")) / "spam_transpiled.qpy"
            assert spam_qpy.exists(), "Call transpile_spam_for_backend_to_isa(backend) before submit_batch_ibm()."
            with open(spam_qpy, "rb") as f:
                spam_isa = load(f)
            n_spam = len(spam_isa)
            assert len(spam_isa[0].cregs) > 0, "SPAM ISA circuits have no classical registers!"

        # ── Accumulate circuits + compute offsets ─────────────────────────────
        all_circuits     = []
        chunk_offsets    = {}
        n_circ_per_chunk = None

        transpiled = {}
        for nc in batch_chunk_ids:
            isa_circuits = self.transpile_chunk_for_backend_to_isa(nc, backend)
            transpiled[nc] = dict(self._last_transpile)
            assert len(isa_circuits[0].cregs) > 0, f"Chunk {nc} ISA circuits have no classical registers!"
            if n_circ_per_chunk is None:
                n_circ_per_chunk = len(isa_circuits)
                from source.core.qpu.layout_selection import final_physical_qubits
                print(f"  [Batch] Circuit 0: depth={isa_circuits[0].depth()}, "
                      f"gates={dict(isa_circuits[0].count_ops())}, "
                      f"layout={final_physical_qubits(isa_circuits[0])}")
            spam_start = len(all_circuits)
            all_circuits += spam_isa + isa_circuits
            circ_start  = spam_start + n_spam
            circ_end    = circ_start + n_circ_per_chunk
            chunk_offsets[nc] = {
                "spam_start": spam_start,
                "circ_start": circ_start,
                "circ_end":   circ_end,
            }

        print(f"  [Batch {batch_chunk_ids[0]}–{batch_chunk_ids[-1]}] "
              f"Total circuits: {len(all_circuits)} "
              f"({len(batch_chunk_ids)} chunks × {n_spam + n_circ_per_chunk})")

        # ── Submit one job ────────────────────────────────────────────────────
        try:
            job, submit_s = self._timed_submit(f"chunk{batch_chunk_ids[0]}-{batch_chunk_ids[-1]}",
                                               lambda: sampler.run(all_circuits, shots=self.CONFIG["shots"]))
        except ChunkSubmitError as e:
            self._write_submission_failed_records(batch_chunk_ids, IBM_CLOUD, backend, e, transpiled, n_spam, n_circ_per_chunk)
            raise
        job_id       = job.job_id()
        submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        print(f"  [Batch {batch_chunk_ids[0]}–{batch_chunk_ids[-1]}] "
              f"Submitted — job_id: {job_id}")

        # ── Save one job record per chunk ─────────────────────────────────────
        for nc in batch_chunk_ids:
            record_path = Path(get_run_subfolder(self.CONFIG, "job_records")) / f"job_record_chunk_{nc}.json"
            record = {
                "backend_name":       str(backend),
                "service_name":       IBM_CLOUD,
                "chunk_id":           nc,
                "job_id":             job_id,
                "submitted_at":       submitted_at,
                "finished_at":        "N/A",
                "nshots":             self.CONFIG["shots"],
                "n_spam":             n_spam,
                "n_circuits":         n_circ_per_chunk,
                "submitted_circuit_files": self._submitted_circuit_files(nc, n_spam),
                # submit_s: the one submission of the whole batched job (shared by its chunks)
                "timing":             {**submit_timing(**transpiled[nc], submit_s=submit_s),
                                       "submit_shared_by_chunks": list(batch_chunk_ids)},
                "submit_attempts":    self._last_submit_attempts,
                "spam_start":         chunk_offsets[nc]["spam_start"],
                "circ_start":         chunk_offsets[nc]["circ_start"],
                "circ_end":           chunk_offsets[nc]["circ_end"],
                "status":             "submitted",
                "result_path":        None,
                "trotter_steps_list": [int(x) for x in self.trotter_steps_list],
                "roles":              self.roles,
                "nsites":             self.CONFIG["nsites"],
                "CONFIG":             saved_config(self.CONFIG)
            }
            archive_failed_record(record_path.parent, nc)   # keep a failed attempt's record (and its cost)
            write_job_record(record_path, record)
            print(f"  [Chunk {nc}] Record saved → {rel_path(record_path)}")

        # ── Free memory ───────────────────────────────────────────────────────
        del all_circuits, isa_circuits
        if self.CONFIG["SPAM_correction"]: del spam_isa
        gc.collect()


    def _circuits_per_chunk(self):
        """Circuits one chunk contributes to a job: its SPAM circuits (one per basis state) + its own."""
        return (2**self.CONFIG["nsites"] if self.CONFIG["SPAM_correction"] else 0) + self.chunk_size

    def _chunks_per_iqm_job(self, max_circuits_per_job):
        """How many whole chunks (each with its own SPAM circuits) fit into one IQM job
        of at most max_circuits_per_job circuits."""
        n_spam      = 2**self.CONFIG["nsites"] if self.CONFIG["SPAM_correction"] else 0   # one SPAM circuit per basis state
        per_chunk   = n_spam + self.chunk_size
        n_per_job   = max_circuits_per_job // per_chunk
        assert n_per_job >= 1, (
            f"One chunk ({n_spam} SPAM + {self.chunk_size} circuits = {per_chunk}) does not fit "
            f"into a job of at most {max_circuits_per_job} circuits.")
        return n_per_job

    def submit_batch_iqm(self, batch_chunk_ids, backend, service_name, max_circuits_per_job):
        """
        Transpile the chunks batch_chunk_ids and submit them as ONE IQM job
        (service_name "MQSS_LRZ", "IQM_Resonance" or "QAAS_IT4I"). Each chunk contributes its
        n_spam SPAM circuits followed by its n_circuits circuits, as in submit_batch_ibm().
        Saves one job record per chunk, all with the same job_id, and the
        spam_start/circ_start/circ_end offsets of the chunk inside the job.

        EQE1 (MQSS) gets no_modify=True so the server keeps our transpilation. IQM
        Resonance's IQMBackend.run() has no such option (iqm-client drops unknown options
        with a warning) and sends the circuits as they are. VLQ's qaas backend uploads the
        circuits as OpenQASM 3 (one HTTP request each) and runs them as they are; its job id
        is the HEAppE job id (an int).
        """
        from qiskit.qpy import load

        assert service_name in (MQSS_LRZ, IQM_RESONANCE, QAAS_IT4I), f"Unknown IQM service: {service_name!r}"
        job_records_dir = Path(get_run_subfolder(self.CONFIG, "job_records"))

        # ── Idempotency — submit only the chunks not yet submitted ────────────
        todo = []
        for nc in batch_chunk_ids:
            record_path = job_records_dir / f"job_record_chunk_{nc}.json"
            if record_path.exists() and json.loads(record_path.read_text(encoding="utf-8"))["status"] in ("submitted", "complete"):
                print(f"  [Chunk {nc}] Already submitted — skipping.")
                continue
            todo.append(nc)
        if not todo:
            return

        # ── Load SPAM ISA circuits once ───────────────────────────────────────
        spam_isa = []
        if self.CONFIG["SPAM_correction"]:
            spam_qpy = Path(get_run_subfolder(self.CONFIG, "transpiled_circuits")) / "spam_transpiled.qpy"
            assert spam_qpy.exists(), "Call transpile_spam_for_backend_to_isa(backend) first."
            with open(spam_qpy, "rb") as f:
                spam_isa = load(f)
            assert len(spam_isa[0].cregs) > 0, "SPAM ISA circuits have no classical registers!"
        n_spam = len(spam_isa)

        # ── Accumulate circuits + compute offsets ─────────────────────────────
        from source.core.qpu.layout_selection import final_physical_qubits
        all_circuits, offsets, transpiled, n_circuits = [], {}, {}, {}
        for nc in todo:
            isa_circuits = self.transpile_chunk_for_backend_to_isa(nc, backend)
            transpiled[nc] = dict(self._last_transpile)
            assert len(isa_circuits[0].cregs) > 0, f"Chunk {nc} ISA circuits have no classical registers!"
            print(f"  [Chunk {nc}] Circuit 0: depth={isa_circuits[0].depth()}, "
                  f"gates={dict(isa_circuits[0].count_ops())}, layout={final_physical_qubits(isa_circuits[0])}")
            spam_start     = len(all_circuits)
            all_circuits  += spam_isa + isa_circuits
            n_circuits[nc] = len(isa_circuits)
            offsets[nc]    = {"spam_start": spam_start,
                              "circ_start": spam_start + n_spam,
                              "circ_end":   spam_start + n_spam + len(isa_circuits)}
        n_in_job = len(all_circuits)
        assert n_in_job <= max_circuits_per_job, (
            f"IQM job of chunks {todo} has {n_in_job} circuits > limit {max_circuits_per_job}")
        label = f"chunk{todo[0]}" if len(todo) == 1 else f"chunk{todo[0]}-{todo[-1]}"
        print(f"  [{service_name}] Job {label}: {len(todo)} chunk(s), {n_in_job} circuits "
              f"(limit {max_circuits_per_job})")

        # ── Submit one job ────────────────────────────────────────────────────
        if service_name == MQSS_LRZ:
            run = lambda: backend.run(all_circuits, shots=self.CONFIG["shots"], no_modify=True)
        elif service_name == QAAS_IT4I:
            print(f"  [IQM_VLQ] Uploading {n_in_job} circuits (~{0.3 * n_in_job / 60:.0f} min at ~0.3 s each)...", flush=True)
            run = lambda: VLQ.run(backend, all_circuits, f"{label}, {n_in_job} circuits",
                                  shots=self.CONFIG["shots"], walltime_limit=VLQ.WALLTIME_LIMIT_S)
        else:
            run = lambda: backend.run(all_circuits, shots=self.CONFIG["shots"])   # async
        try:
            job, submit_s = self._timed_submit(label, run)
        except ChunkSubmitError as e:
            self._write_submission_failed_records(todo, service_name, backend, e, transpiled, n_spam, n_circuits)
            raise
        job_id        = job.job_id()
        submitted_at  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        upload_end_at = now_utc_iso(precise=True)
        print(f"  [{service_name}] Job {label} submitted — job_id: {job_id}")

        # ── Save one job record per chunk ─────────────────────────────────────
        for nc in todo:
            record_path = job_records_dir / f"job_record_chunk_{nc}.json"
            record = {
                "backend_name":       getattr(backend, "name", None) or str(backend),
                "service_name":       service_name,
                "chunk_id":           nc,
                "job_id":             job_id,
                "submitted_at":       submitted_at,
                "upload_end_at":      upload_end_at,   # precise end of the submission call (VLQ: its queue starts here)
                "scheduled_at":       "N/A",         # filled in by collect
                "finished_at":        "N/A",
                "nshots":             self.CONFIG["shots"],
                "n_spam":             n_spam,
                "n_circuits":         n_circuits[nc],
                "n_circuits_in_job":  n_in_job,
                **offsets[nc],
                "submitted_circuit_files": self._submitted_circuit_files(nc, n_spam),
                # submit_s: the one submission of the whole job (shared by its chunks)
                "timing":             {**submit_timing(**transpiled[nc], submit_s=submit_s),
                                       "submit_shared_by_chunks": list(todo)},
                "submit_attempts":    self._last_submit_attempts,
                "status":             "submitted",
                "result_path":        None,
                "trotter_steps_list": [int(x) for x in self.trotter_steps_list],
                "roles":              self.roles,
                "nsites":             self.CONFIG["nsites"],
                "CONFIG":             saved_config(self.CONFIG),
            }
            archive_failed_record(job_records_dir, nc)   # keep a failed attempt's record (and its cost)
            write_job_record(record_path, record)
            print(f"  [Chunk {nc}] Record saved → {rel_path(record_path)}")

        del all_circuits
        gc.collect()

    def _write_submission_failed_records(self, chunk_ids, service_name, backend, err, transpiled, n_spam, n_circuits):
        """status 'failed' records for the chunks of a job whose submission did not succeed
        (err: ChunkSubmitError from _timed_submit). One record per chunk, sharing a
        submission_id so the analysis counts the job once. may_exist_on_provider: the answer
        was lost after the request may have arrived — check the provider for the job before
        resubmitting. A previous failed record of a chunk is archived first. Never raises —
        the caller re-raises err."""
        import uuid
        job_records_dir = Path(get_run_subfolder(self.CONFIG, "job_records"))
        may_exist       = err.kind == "answer_lost"
        submission_id   = f"failed-submission-{uuid.uuid4().hex[:12]}"
        submitted_at    = err.attempts[0]["at"] if err.attempts else now_utc_iso()
        for nc in chunk_ids:
            try:
                current = job_records_dir / f"job_record_chunk_{nc}.json"
                if current.exists() and json.loads(current.read_text(encoding="utf-8")).get("status") in ("submitted", "complete"):
                    continue   # a chunk already on the QPU (submit() called with it in a batch) keeps its record
                record = {
                    "chunk_id":              nc,
                    "status":                "failed",
                    "failure":               scrub_paths(f"submission failed: {err.cause!r}")[:2000],
                    "failure_stage":         "submission",
                    "may_exist_on_provider": may_exist,
                    "service_name":          service_name,
                    "backend_name":          getattr(backend, "name", None) or str(backend),
                    "job_id":                None,
                    "submission_id":         submission_id,
                    "submitted_at":          submitted_at,
                    "scheduled_at":          "N/A",
                    "finished_at":           "N/A",
                    "nshots":                self.CONFIG["shots"],
                    "n_spam":                n_spam,
                    "n_circuits":            n_circuits[nc] if isinstance(n_circuits, dict) else n_circuits,
                    "submit_attempts":       err.attempts,
                    "timing":                {**submit_timing(**transpiled[nc], submit_s=round(err.submit_s, 3)),
                                              "qpu_s": 0.0, "qpu_s_source": "not executed",
                                              "queue_s": None, "queue_s_source": "unavailable",
                                              **({"submit_shared_by_chunks": list(chunk_ids)} if len(chunk_ids) > 1 else {})},
                    "result_path":           None,
                    "CONFIG": saved_config(self.CONFIG),
                }
                archive_failed_record(job_records_dir, nc)
                write_job_record(job_records_dir / f"job_record_chunk_{nc}.json", record)
            except Exception as e:
                print(f"  ⚠️  [Chunk {nc}] Could not write the submission-failed record: {e!r}")
        print(f"  ❌ Submission of chunk(s) {format_chunk_range(list(chunk_ids))} failed after "
              f"{len(err.attempts)} attempt(s): {err.cause!r} — recorded as failed (job_records/).", file=sys.stderr)
        if may_exist:
            print("  ⚠️  The connection broke after the request may have reached the provider: the job may exist "
                  "there and use QPU time. Check the provider (its dashboard / check_real_qpu_job_status_*.py) "
                  "before resubmitting these chunks.")

    def _write_aqt_failed_record(self, record_path, nc, backend, n_spam, n_circuits, transpiled, exc, *,
                                 submitted_at, wall_s, batch_walls, batch_sizes, shots_p,
                                 failed_batch_s, n_done_in_failed_batch=0, part, shot_parts):
        """Job record of an AQT chunk attempt that raised while a shot part ran: status 'failed',
        with the QPU time of the part's finished batches and of the n_done_in_failed_batch circuits
        that finished inside the failing batch (lost: only whole parts are kept); the failing batch's
        wall time is a batch of its own, so its remainder after that QPU time is queue time.
        A previous failed record is archived first. Never raises — the caller re-raises the original error."""
        try:
            k       = int(n_done_in_failed_batch)
            sizes   = list(batch_sizes) + [k]
            timing = {**submit_timing(**transpiled),
                      **aqt_timing(list(batch_walls) + [max(float(failed_batch_s), 0.0)], sizes, self.CONFIG["shots"],
                                   batch_shots=[shots_p] * len(sizes))}
            timing["failed_batch_s"]  = round(float(failed_batch_s), 3)
            stage  = getattr(exc, "aqt_failure_stage", None) or "execution"
            record = {
                "backend_name":        str(backend),
                "service_name":        PCSS,
                "chunk_id":            nc,
                "job_id":              None,
                "failure_stage":       stage,
                "may_exist_on_provider": False,   # synchronous: nothing of this attempt keeps running
                "submitted_at":        submitted_at,
                "scheduled_at":        "N/A",
                "finished_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "wall_seconds":        round(float(wall_s), 3),
                "nshots":              self.CONFIG["shots"],
                "shot_parts":          shot_parts,
                "failed_part":         part,
                "n_spam":              n_spam,
                "n_circuits":          n_circuits,
                "n_executed_circuits": sum(sizes),
                "n_executed_shots":    sum(sizes) * shots_p,
                "timing":              timing,
                "status":              "failed",
                "failure":             scrub_paths(f"submission failed (shot part {part}): {exc!r}" if stage == "submission"
                                                   else f"shot part {part} raised: {exc!r}")[:2000],
                "result_path":         None,
                "CONFIG": saved_config(self.CONFIG),
            }
            archive_failed_record(record_path.parent, nc)
            write_job_record(record_path, record)
            print(f"  ⚠️  [Chunk {nc}] Failed in shot part {part + 1}/{len(shot_parts)} ({stage}) — recorded as failed → {record_path.name}\n"
                  f"      {type(exc).__name__}: {' '.join(str(exc).split())[:80]}")
        except Exception as e:
            print(f"  ⚠️  [Chunk {nc}] Could not write the failed record: {e!r}")

    def submit_chunk_aqt(self, nc, backend):
        from qiskit.qpy import load

        record_path = Path(get_run_subfolder(self.CONFIG, "job_records")) / f"job_record_chunk_{nc}.json"

        # ── Idempotency ─────────────────────────────────────────────────────── 
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["status"] in ("submitted", "complete"):
                print(f"  [Chunk {nc}] Already submitted — skipping.")
                return

        # ── Transpile circuits ────────────────────────────────────────────────
        # AQT runs all_to_all, so this loads chunk_{nc}_qiskit_v13.qpy (QPY version 13),
        # which aqt.venv's qiskit==1.4.6 can read — see readme_aqt.md §1.
        isa_circuits = self.transpile_chunk_for_backend_to_isa(nc, backend)
        from source.core.qpu.layout_selection import final_physical_qubits
        print(f"  [Chunk {nc}] Circuit 0: depth={isa_circuits[0].depth()}, "
              f"gates={dict(isa_circuits[0].count_ops())}, layout={final_physical_qubits(isa_circuits[0])}")

        final_qubits = final_physical_qubits(isa_circuits[0])
        layout_dict  = {str(v): p for v, p in enumerate(final_qubits)} if final_qubits else "N/A"

        # ── Add SPAM circuits if enabled ──────────────────────────────────────
        if self.CONFIG["SPAM_correction"]:
            spam_qpy = Path(get_run_subfolder(self.CONFIG, "transpiled_circuits")) / "spam_transpiled.qpy"
            assert spam_qpy.exists(), "Call transpile_spam_for_backend_to_isa(backend) first."
            with open(spam_qpy, "rb") as f: 
                spam_isa = load(f)
            all_circuits = spam_isa + isa_circuits
            n_spam       = len(spam_isa)
            assert len(spam_isa[0].cregs) > 0,    "SPAM ISA circuits have no classical registers!"
            assert len(isa_circuits[0].cregs) > 0, "ISA circuits have no classical registers!"
        else:
            all_circuits = isa_circuits
            n_spam       = 0
            assert len(isa_circuits[0].cregs) > 0, "ISA circuits have no classical registers!"

        # ── Deterministic QPU-time estimate (queue-independent) ───────────────
        shots = self.CONFIG["shots"]
        try:
            qpu_us_main = sum(aqt_circuit_qpu_us(c) for c in isa_circuits) * shots
            qpu_us_spam = (sum(aqt_circuit_qpu_us(c) for c in spam_isa) * shots
                           if self.CONFIG["SPAM_correction"] else 0.0)
        except ValueError as e:
            print(f"  [Chunk {nc}] QPU-time estimate skipped: {e}")
            qpu_us_main = qpu_us_spam = None

        # ── Submit: shot parts x batches of 50 (qiskit-aqt-provider limit) ─────
        # The qiskit-aqt-provider client rejects a job of more than 50 circuits (its payload model, not an
        # AQT/PCSS service limit). AQT runs at most AQT_MAX_SHOTS shots per circuit, so a larger shot count is split
        # into the fewest, most even parts (aqt_shot_parts: 600 -> 3 x 200). Each part runs
        # the whole chunk (SPAM + main) and is checkpointed to raw_results/shot_parts/, so an
        # interrupted chunk resumes at the first missing part. The parts' counts are summed
        # per circuit below, so everything downstream sees nshots shots per circuit.
        MAX_AQT_BATCH = 50
        transpiled    = dict(self._last_transpile)
        shot_parts    = aqt_shot_parts(shots)
        parts_dir     = Path(get_run_subfolder(self.CONFIG, "raw_results")) / "shot_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        if len(shot_parts) > 1:
            print(f"  [Chunk {nc}] {shots} shots > AQT's {AQT_MAX_SHOTS} per circuit — "
                  f"running the chunk {len(shot_parts)}× with shots {shot_parts}.")

        parts = []
        for p, shots_p in enumerate(shot_parts):
            part_path = parts_dir / f"chunk_{nc}_part_{p}.json"
            if part_path.exists():
                part = json.loads(part_path.read_text(encoding="utf-8"))
                if (part.get("shot_parts") == shot_parts and part.get("shots") == shots_p
                        and len(part.get("counts", [])) == len(all_circuits)):
                    print(f"  [Chunk {nc}] Shot part {p + 1}/{len(shot_parts)} ({shots_p} shots) "
                          f"already on disk (finished {part['finished_at']}) — reusing it.")
                    part["resumed"] = True
                    parts.append(part)
                    continue
                print(f"  ⚠️  [Chunk {nc}] Shot part file {part_path.name} doesn't match this chunk "
                      f"(shots/circuits differ) — running the part again.")

            # Parts reused from disk above log nothing; each part run here is bracketed like a chunk.
            self._log_timestamp(f"qpu_chunk{nc}_part{p}_start")
            part_submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            part_t0           = time.perf_counter()
            part_results      = []   # flat list of count dicts, in submission order
            batch_walls, batch_sizes = [], []   # per backend.run(): submit -> result wall time (queue + QPU)
            b_t0              = part_t0
            try:
                for batch_start in range(0, len(all_circuits), MAX_AQT_BATCH):
                    batch      = all_circuits[batch_start : batch_start + MAX_AQT_BATCH]
                    batch_end  = batch_start + len(batch)
                    print(f"  [Chunk {nc}] Part {p + 1}/{len(shot_parts)}: submitting circuits "
                          f"{batch_start}–{batch_end - 1} ({len(batch)} circuits, {shots_p} shots)...")
                    b_t0       = time.perf_counter()
                    result     = _aqt_run(backend, batch, shots_p).result()
                    batch_walls.append(time.perf_counter() - b_t0)
                    batch_sizes.append(len(batch))
                    batch_counts = result.get_counts()
                    # get_counts() returns a list when multiple circuits, a single dict when one
                    if isinstance(batch_counts, dict):
                        batch_counts = [batch_counts]
                    part_results.extend(batch_counts)
            except Exception as exc:
                # The part's finished batches and the circuits finished inside the failing batch ran on
                # the QPU but their counts are lost (only whole parts are checkpointed): record what
                # this attempt cost, then fail.
                self._write_aqt_failed_record(
                    record_path, nc, backend, n_spam, len(isa_circuits), transpiled, exc,
                    submitted_at=part_submitted_at, wall_s=time.perf_counter() - part_t0,
                    batch_walls=batch_walls, batch_sizes=batch_sizes, shots_p=shots_p,
                    failed_batch_s=time.perf_counter() - b_t0,
                    n_done_in_failed_batch=getattr(exc, "aqt_circuits_done", 0), part=p, shot_parts=shot_parts)
                raise

            assert len(part_results) == len(all_circuits), (
                f"Result count mismatch in part {p}: got {len(part_results)}, expected {len(all_circuits)}"
            )
            part = {
                "chunk_id":           nc,
                "part":               p,
                "shot_parts":         shot_parts,
                "shots":              shots_p,
                "submitted_at":       part_submitted_at,
                "finished_at":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "wall_seconds":       round(time.perf_counter() - part_t0, 3),
                "batch_wall_seconds": batch_walls,
                "batch_sizes":        batch_sizes,
                "counts":             [dict(c) for c in part_results],   # raw AQT bit order, submission order
            }
            tmp = part_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(part), encoding="utf-8")
            os.replace(tmp, part_path)                   # atomic: a crash never leaves a half-written part
            self._log_timestamp(f"qpu_chunk{nc}_part{p}_finished")
            part["resumed"] = False
            parts.append(part)

        # ── Results received: their handling (merging parts, saving) is the chunk's collection ──
        self._log_timestamp(f"chunk{nc}_collect_start")
        collect_t0 = time.perf_counter()

        # ── Merge the parts: sum each circuit's counts ────────────────────────
        all_results = []
        for ci in range(len(all_circuits)):
            merged = {}
            for part in parts:
                for k, v in part["counts"][ci].items():
                    merged[k] = merged.get(k, 0) + int(v)
            assert sum(merged.values()) == shots, (
                f"Chunk {nc}, circuit {ci}: merged parts hold {sum(merged.values())} shots, expected {shots}"
            )
            all_results.append(merged)

        batch_walls  = [w for part in parts for w in part["batch_wall_seconds"]]
        batch_sizes  = [n for part in parts for n in part["batch_sizes"]]
        batch_shots  = [part["shots"] for part in parts for _ in part["batch_sizes"]]
        wall_seconds = sum(part["wall_seconds"] for part in parts)
        submitted_at = parts[0]["submitted_at"]
        finished_at  = parts[-1]["finished_at"]

        # ── Split merged results into SPAM and main circuits
        spam_raw    = [all_results[si] for si in range(n_spam)]
        all_counts  = [all_results[n_spam + ci] for ci in range(len(isa_circuits))]

        # spam_raw   = [result.get_counts(all_circuits[si]) for si in range(n_spam)]
        # all_counts = [result.get_counts(all_circuits[n_spam + ci]) for ci in range(len(isa_circuits))]

        # Reverse bitstrings
        spam_raw   = [{k[::-1]: v for k, v in c.items()} for c in spam_raw]
        all_counts = [{k[::-1]: v for k, v in c.items()} for c in all_counts]       

        # ── Build measurement_counts and fidelity_nec_dict ────────────────────
        trotter_steps_list = [int(round(T / self.CONFIG["dt"])) for T in self.CONFIG["T_list"]]
        roles              = self.roles
        nsites             = self.CONFIG["nsites"]
        chunk_key          = f"chunk_{nc}"
        measurement_counts = {}
        fidelity_nec_dict  = {}

        for di, nt in enumerate(trotter_steps_list):
            tag = f"Nt_{nt:02d}"
            for ri, role in enumerate(roles):
                # local_idx: ranges 0 to n_depths * len(self.roles) - 1
                # identical across all chunks — same keys appear in every chunk_{nc}
                local_idx = di * len(roles) + ri
                job_key   = f"job_{tag}_{role}_ems{local_idx}"
                circ_key  = f"{tag}_{role}_circuit_0" # always 0 — sample index inside a chunk (1 sample per chunk)
                counts    = all_counts[local_idx]
                if role.endswith("_nec"):
                    fidelity_nec_dict[job_key]  = {circ_key: counts}
                else:
                    measurement_counts[job_key] = {circ_key: counts}

        # ── Save counts and fidelity ──────────────────────────────────────────
        (Path(get_run_subfolder(self.CONFIG, "raw_results"))).mkdir(parents=True, exist_ok=True)
        counts_path   = Path(get_run_subfolder(self.CONFIG, "raw_results")) / "results_counts.json"
        fidelity_path = Path(get_run_subfolder(self.CONFIG, "raw_results")) / "results_fidelity.json"

        existing_counts   = json.loads(counts_path.read_text(encoding="utf-8"))   if counts_path.exists()   else {}
        existing_fidelity = json.loads(fidelity_path.read_text(encoding="utf-8")) if fidelity_path.exists() else {}

        existing_counts[chunk_key]   = measurement_counts
        existing_fidelity[chunk_key] = fidelity_nec_dict

        with open(counts_path, "w", encoding="utf-8") as f:   json.dump(existing_counts,   f, indent=4)
        with open(fidelity_path, "w", encoding="utf-8") as f: json.dump(existing_fidelity, f, indent=4)

        # ── Save SPAM counts ──────────────────────────────────────────────────
        if self.CONFIG["SPAM_correction"]:
            spam_path     = Path(get_run_subfolder(self.CONFIG, "raw_results")) / "spam_calibration_counts_per_chunk.json"
            existing_spam = json.loads(spam_path.read_text(encoding="utf-8")) if spam_path.exists() else {}
            existing_spam[chunk_key] = {
                format(si, f"0{nsites}b"): spam_raw[si] for si in range(n_spam)
            }
            with open(spam_path, "w", encoding="utf-8") as f: json.dump(existing_spam, f, indent=4)

        print_chunk_saved(nc, n_spam, len(measurement_counts), len(fidelity_nec_dict))

        # ── Save job record as complete ───────────────────────────────────────
        qpu_s_physics = None if qpu_us_main is None else round((qpu_us_main + qpu_us_spam) / 1e6, 6)
        record = {
            "backend_name":              str(backend),
            "service_name":              PCSS,
            "layout":                    layout_dict,
            "chunk_id":                  nc,
            "submitted_circuit_files":   self._submitted_circuit_files(nc, n_spam),
            "timing":                    {**submit_timing(**transpiled),
                                          **aqt_timing(batch_walls, batch_sizes, self.CONFIG["shots"], qpu_s_physics_model=qpu_s_physics,
                                                       batch_shots=batch_shots)},
            "circuit_check":             CC.not_available("AQT does not return the executed circuits"),
            "submitted_at":              submitted_at,
            "scheduled_at":              "N/A",
            "finished_at":               finished_at,
            "wall_seconds":              round(wall_seconds, 3),
            "qpu_seconds_estimate":      qpu_s_physics,
            "qpu_seconds_estimate_main": None if qpu_us_main is None else round(qpu_us_main / 1e6, 6),
            "qpu_seconds_estimate_spam": None if qpu_us_spam is None else round(qpu_us_spam / 1e6, 6),
            "nshots":                    self.CONFIG["shots"],                       # total per circuit, all parts
            "shot_parts":                shot_parts,                                 # shots of each execution of the chunk
            "max_shots_per_circuit":     AQT_MAX_SHOTS,
            "n_executed_circuits":       len(all_circuits) * len(shot_parts),
            "shot_part_times":           [{"part": pi, "shots": part["shots"], "submitted_at": part["submitted_at"],
                                           "finished_at": part["finished_at"], "wall_seconds": part["wall_seconds"],
                                           "resumed": part["resumed"]} for pi, part in enumerate(parts)],
            "n_spam":                    n_spam,
            "n_circuits":                len(isa_circuits),
            "status":                    "complete",
            "result_path":               run_relative_path(self.CONFIG["folder_path"], "raw_results", "results_counts.json"),
            "trotter_steps_list":        trotter_steps_list,
            "roles":                     roles,
            "nsites":                    nsites,
            "CONFIG": saved_config(self.CONFIG)
        }
        record["timing"]["collect_s"] = round(time.perf_counter() - collect_t0, 3)
        archive_failed_record(record_path.parent, nc)   # keep a failed attempt's record (and its cost)
        write_job_record(record_path, record)
        self._log_timestamp(f"chunk{nc}_collect_end")
        print(f"  [Chunk {nc}] Job record saved → {rel_path(record_path)}")
        del isa_circuits, all_circuits, all_results, parts
        if self.CONFIG["SPAM_correction"]: del spam_isa
        gc.collect()

    def _load_raw_chunk_qiskit(self, nc):
        """Raw (untranspiled) Qiskit circuits of chunk nc, with measurements.
        Loads the build format's chunk_{nc}_{kind}.qpy (kind "qiskit" for LNN,
        "qiskit_v13" for all_to_all) if present, else converts from myQLM."""
        from qiskit.qpy import load

        chunk_kind   = "qiskit_v13" if self._qiskit_v13_qpy else "qiskit"
        circ_top     = get_circuit_folder(self.CONFIG, top_folder=True)
        migrate_flat_chunk_files(circ_top)
        qpy_raw_path = Path(get_chunk_folder(circ_top, chunk_kind)) / f"chunk_{nc}_{chunk_kind}.qpy"

        if qpy_raw_path.exists():
            print(f"  [Chunk {nc}] Raw Qiskit circuits found — skipping myQLM conversion.")
            with open(qpy_raw_path, "rb") as f:
                qk_circuits = load(f)
        else:
            qk_circuits = self.convert_myqlm_chunk_to_qiskit(nc)

        for qk_circ in qk_circuits:
            if not qk_circ.cregs:
                qk_circ.measure_all()
        return qk_circuits

    def record_run_info(self, run_folder):
        """
        Write datafiles/<run_folder>/run_info.json now, before submit() would.

        Called before the pre-submission health check (directly, or via
        resolve_physical_qubits(run_folder=...)), so a run aborted because that
        check found the QPU degraded is still recorded: run_post_process.py then
        analyzes its health checks and reports that the benchmark was not run.
        submit() rewrites the same file only if something differs.
        """
        assert self._status == "built", "record_run_info() needs build() first — it records the built configuration."
        if self.CONFIG.get("folder_path") is None:
            self.CONFIG["folder_path"] = os.path.join(PROJECT_ROOT, "datafiles", run_folder)
        write_run_info(self.CONFIG, os.path.join(PROJECT_ROOT, "datafiles", run_folder),
                       self.CONFIG["lnn_architecture"], circuits_qiskit_v13_qpy=self._qiskit_v13_qpy,
                       instance=self.run_instance[0], instance_plan=self.run_instance[1])

    def resolve_physical_qubits(self, backend, physical_qubits=None, run_folder=None, layout=None,
                                layout_backend=None, calibration_source=None):
        """
        Fix the physical qubits every chunk and the SPAM batch of this run are
        transpiled onto, and record them in transpiled_circuits/physical_qubits.json.

        Called at the start of submit(), before any transpilation. The real-QPU
        submit scripts also call it right after build(), before their first
        health check, so that health check runs on the run's qubits too
        (submit_health_check reads the record). On a resumed run the recorded
        choice is reused, so a run never changes qubits halfway. See
        source/core/qpu/layout_selection.py for the layouts and the selection criterion.

        :param backend:            backend object with a .target — the QPU the run is transpiled for
        :param layout:             what to choose (source/core/runner_support/benchmark_args.py LayoutArgs): .mode optimized | random |
                                   lowest | list (.qubits). None: optimized, or the list `physical_qubits`
        :param physical_qubits:    optional list[int] — shorthand for layout=list of these qubits
        :param layout_backend:     backend whose Target the layouts are scored / enumerated on, when it is not the QPU's
                                   own — a calibration file's offline backend (source/core/qpu/calibration_input.py). The
                                   choice is then checked against `backend`'s Target, and refused if it cannot run it.
        :param calibration_source: the calibration file's name, recorded (never an absolute path)
        :param run_folder:         str — subfolder under datafiles/; needed only when
                                   called before submit() (same meaning as submit's)
        :returns:                  list[int] | None — the initial_layout to use
        """
        from source.core.qpu.layout_selection import choose_layout, check_forced_layout, placement_filter

        if run_folder is not None:
            self.record_run_info(run_folder)

        mode   = getattr(layout, "mode", None) or ("list" if physical_qubits is not None else "optimized")
        forced = getattr(layout, "qubits", None) or physical_qubits
        tdir        = Path(get_run_subfolder(self.CONFIG, "transpiled_circuits"))
        record_path = tdir / "physical_qubits.json"

        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if forced is not None and mode == "list" and list(forced) != record["physical_qubits"]:
                raise ValueError(
                    f"physical_qubits={list(forced)} differs from this run's recorded "
                    f"{record['physical_qubits']} ({record_path}) — a run must stay on one set of qubits."
                )
            if (layout is not None or physical_qubits is not None) and record.get("layout_mode") not in (None, mode):
                raise ValueError(
                    f"layout={mode} but this run's recorded layout is {record['layout_mode']} "
                    f"{record['physical_qubits']} ({record_path}) — a run must stay on one set of qubits."
                )
            self._physical_qubits = record["physical_qubits"]
            print(f"  [Layout] Using recorded physical qubits {self._physical_qubits} ({record['method']}).")
            return self._physical_qubits

        # A run that already transpiled circuits before this record existed was
        # laid out per circuit by qiskit; pinning would mix two layouts in one run.
        if any(tdir.glob("chunk_*_transpiled.qpy")) or (tdir / "spam_transpiled.qpy").exists():
            print(f"  ⚠️  [Layout] {rel_path(tdir)} already holds transpiled circuits but no physical_qubits.json "
                  f"— legacy run, keeping qiskit's per-circuit layout (not pinned).")
            self._physical_qubits = None
            return None

        score_backend = layout_backend or backend
        target  = getattr(score_backend, "target", None)
        vlq_cal = getattr(backend, "vlq_calibration", None)   # set by the VLQ scripts (source/core/qpu/iqm_vlq.py)
        if target is None and mode == "list":   # a forced list on a backend that describes no Target: nothing to check it against
            record = dict(physical_qubits=list(forced), method="user", layout_mode=mode,
                          log_success=None, top_candidates=[], n_candidates=None, gates=None)
        elif target is None:
            record = dict(physical_qubits=None, method="no_target", layout_mode=mode,
                          log_success=None, top_candidates=[], n_candidates=0, gates=None)
        else:
            # Choose on chunk 0's deepest circuit (most 2q gates) — all chunks share
            # the same structure up to single-qubit gates (randomized compiling).
            raw = self._load_raw_chunk_qiskit(0)
            rep = max(raw, key=lambda c: sum(1 for i in c.data if len(i.qubits) == 2))
            # Layouts chosen on a calibration file's Target (not the QPU's own) only consider what the QPU has too:
            # the file may be older and describe qubits / couplers the live backend no longer lists.
            live_ok = (placement_filter(rep, backend.target)
                       if layout_backend is not None and getattr(backend, "target", None) is not None else None)
            if vlq_cal is not None:   # IQM star: each virtual qubit's prx/MOVE/CZ roles, graded from VLQ's public calibration
                roles  = VLQ.representative_roles(rep, vlq_cal, backend if VLQ.is_vlq_backend(backend) else None)   # else: offline target
                record = VLQ.choose_vlq_layout(mode, roles, vlq_cal, forced)
            elif mode == "optimized":
                # Native single-qubit gate counts from one transpilation onto any SWAP-free
                # placement (they don't depend on which), then every placement scored with them.
                from source.core.qpu.layout_selection import probe_placement, select_physical_qubits, transpiled_1q_counts
                probe  = probe_placement(rep, target)
                record = select_physical_qubits(
                    rep, target, None if probe is None else transpiled_1q_counts(rep, score_backend, probe), live_ok)
                record["layout_mode"] = mode
            else:
                record = choose_layout(mode, rep, target, forced, placement_ok=live_ok)
            if (mode == "optimized" and layout_backend is None and vlq_cal is None
                    and record["physical_qubits"] is None and record.get("method") == "no_swap_free_placement"):
                # Unpinned, the run would only fail later in _check_transpiled (routing needs SWAPs): refuse it here.
                # Only when the circuits truly have no SWAP-free placement; "no_scored_placement" (placements exist
                # but none has usable calibration data) keeps its earlier behaviour: unpinned, qiskit lays it out.
                raise ValueError(f"layout=optimized: the QPU ({getattr(backend, 'name', backend)}) has no placement of "
                                 f"the circuits without SWAPs (among its usable qubits and couplers). "
                                 f"Pin the qubits explicitly with layout=a,b,c (three distinct qubits).")
            if layout_backend is not None and record["physical_qubits"] is None:
                raise ValueError(f"no layout of the circuits exists that both the calibration "
                                 f"{f'file ({calibration_source})' if calibration_source else 'snapshot of this run'} "
                                 f"and the QPU ({getattr(backend, 'name', backend)}) support (method {record['method']})")
            if layout_backend is not None and record["physical_qubits"] is not None and getattr(backend, "target", None) is not None:
                try:
                    check_forced_layout(rep, backend.target, record["physical_qubits"])
                except ValueError as e:
                    raise ValueError(f"the layout chosen with the calibration {'file' if calibration_source else 'snapshot'} "
                                     f"cannot be run on the QPU itself ({getattr(backend, 'name', backend)}) — is it out "
                                     f"of date? {e}") from None

        record["backend"]        = getattr(backend, "name", str(backend))
        record["selected_at"]    = now_utc_iso(precise=True)
        if calibration_source is not None:
            record["calibration_source"] = calibration_source
        arch                     = getattr(backend, "architecture", None)
        if record.get("calibration_set_id") is None:
            record["calibration_set_id"] = (str(arch.calibration_set_id)
                                            if arch is not None and getattr(arch, "calibration_set_id", None) else None)
        tdir.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=4), encoding="utf-8")

        self._physical_qubits = record["physical_qubits"]
        print(f"  [Layout] Physical qubits {self._physical_qubits} — layout={mode}, method={record['method']}"
              + (f", log P={record['log_success']:.4f}" if record.get("log_success") is not None else "")
              + (f", seed={record['seed']}" if record.get("seed") is not None else "")
              + (f", calibration file {calibration_source}" if calibration_source else "")
              + f" → {rel_path(record_path)}")
        return self._physical_qubits

    def _check_transpiled(self, label, raw_circuits, isa_circuits, check_2q=True):
        """Assert transpilation kept every 2q gate (no SWAPs added, none removed)
        and put every circuit on the same physical qubits — the pinned ones, if any."""
        from source.core.qpu.layout_selection import count_2q_ops, final_physical_qubits

        if check_2q:
            for i, (raw, isa) in enumerate(zip(raw_circuits, isa_circuits)):
                n_raw, n_isa = count_2q_ops(raw), count_2q_ops(isa)
                assert n_raw == n_isa, (
                    f"{label} circuit {i}: 2q gate count changed in transpilation {n_raw} → {n_isa} "
                    f"(routing inserted SWAPs, or gates were cancelled)"
                )

        layouts = {tuple(l) if l is not None else None
                   for l in (final_physical_qubits(c) for c in isa_circuits)}
        assert len(layouts) == 1, f"{label}: circuits ended up on different physical qubits {layouts}"
        (layout,) = layouts
        pinned = getattr(self, "_physical_qubits", None)
        if pinned is not None:
            assert layout == tuple(pinned), (
                f"{label}: transpiled onto {layout}, expected the run's physical qubits {tuple(pinned)}"
            )
        return layout

    def transpile_chunk_for_backend_to_isa(self, nc, backend):
        """
        Transpile chunk nc to the given backend's ISA using qiskit.transpile
        (optimization_level=1), onto the run's physical qubits (resolve_physical_qubits()).
        Idempotent — loads from QPY if already transpiled.

        Used for every backend: IBM (real and fake), IQM EQE1, IQM Resonance,
        AQT, and local simulation with a layout backend.

        :param nc:              int — chunk index
        :param backend:         backend object
        :returns:       list of ISA QuantumCircuits
        """
        from qiskit import transpile
        from qiskit.qpy import dump, load

        qpy_isa_path = Path(get_run_subfolder(self.CONFIG, "transpiled_circuits")) / f"chunk_{nc}_transpiled.qpy"

        # ── Idempotency — load if already transpiled ──────────────────────────
        if qpy_isa_path.exists():
            print(f"  [Chunk {nc}] Transpiled ISA circuits found — loading.")
            self._last_transpile = {"transpile_s": None, "transpile_reused": True}
            with open(qpy_isa_path, "rb") as f:
                return load(f)

        qk_circuits = self._load_raw_chunk_qiskit(nc)

        # ── Transpile (timed: chunk{nc}_transpile_start/_end, and _last_transpile) ──
        self._log_timestamp(f"chunk{nc}_transpile_start")
        t0 = time.perf_counter()
        isa_circuits = transpile(qk_circuits, backend, optimization_level=1,
                                 initial_layout=getattr(self, "_physical_qubits", None),
                                 scheduling_method=scheduling_method(backend))
        if not isinstance(isa_circuits, list):
            isa_circuits = [isa_circuits]
        self._check_transpiled(f"Chunk {nc}", qk_circuits, isa_circuits)

        # ── Save ISA QPY ──────────────────────────────────────────────────────
        with open(qpy_isa_path, "wb") as f:
            dump(drop_identity_final_layout(isa_circuits), f)   # VLQ layouts: see layout_selection
        self._last_transpile = {"transpile_s": time.perf_counter() - t0, "transpile_reused": False}
        self._log_timestamp(f"chunk{nc}_transpile_end")
        print(f"  [Chunk {nc}] Transpiled {len(isa_circuits)} circuits → {rel_path(qpy_isa_path)}")
        export_circuit_drawings(self.CONFIG["folder_path"], chunk_items(self.roles, qk_circuits, isa_circuits))

        del qk_circuits
        gc.collect()

        return isa_circuits


    def transpile_spam_for_backend_to_isa(self, backend):
        """
        Transpile SPAM circuits to the given backend's ISA using qiskit.transpile,
        onto the same physical qubits as the chunks (resolve_physical_qubits()).
        The SPAM circuits aren't part of the chunks — they're prepended to each
        job after the chunk is transpiled — so they need their own ISA copy.
        Idempotent — loads if already transpiled.
        Saves to run_folder/transpiled_circuits/spam_transpiled.qpy.

        :param backend: backend object
        :returns:       list of ISA SPAM circuits
        """
        from qiskit import transpile
        from qiskit.qpy import dump, load

        spam_qpy_isa_path = Path(get_run_subfolder(self.CONFIG, "transpiled_circuits")) / "spam_transpiled.qpy"

        # ── Idempotency ───────────────────────────────────────────────────────
        if spam_qpy_isa_path.exists():
            print(f"  [SPAM] Transpiled SPAM circuits found — loading.")
            with open(spam_qpy_isa_path, "rb") as f:
                return load(f)

        # ── Build raw SPAM circuits ───────────────────────────────────────────
        if not hasattr(self, "_qk_spam_circuits"):
            self.build_spam_batch_qiskit()

        # cregs check is still a safety net in case circuits came from elsewhere
        for qk_circ in self._qk_spam_circuits:
            if not qk_circ.cregs:
                qk_circ.measure_all()

        # ── Transpile (timed: spam_transpile_start/_end) ──────────────────────
        self._log_timestamp("spam_transpile_start")
        t0 = time.perf_counter()
        isa_spam = transpile(self._qk_spam_circuits, backend, optimization_level=1,
                             initial_layout=getattr(self, "_physical_qubits", None),
                             scheduling_method=scheduling_method(backend))
        if not isinstance(isa_spam, list):
            isa_spam = [isa_spam]
        self._check_transpiled("SPAM", self._qk_spam_circuits, isa_spam, check_2q=False)

        # ── Save ──────────────────────────────────────────────────────────────
        with open(spam_qpy_isa_path, "wb") as f:
            dump(drop_identity_final_layout(isa_spam), f)   # VLQ layouts: see layout_selection
        transpile_s = time.perf_counter() - t0
        self._log_timestamp("spam_transpile_end")
        spam_qpy_isa_path.with_suffix(".json").write_text(json.dumps(
            {"transpile_s": transpile_s, "transpiled_at": now_utc_iso(precise=True)}, indent=2), encoding="utf-8")
        print(f"  [SPAM] Transpiled {len(isa_spam)} SPAM circuits → {rel_path(spam_qpy_isa_path)}")
        export_circuit_drawings(self.CONFIG["folder_path"], [("00_SPAM", self._qk_spam_circuits[0], isa_spam[0])])

        return isa_spam

    def _log_timestamp(self, label):
        """timestamp() into this run's log, if one is set (submit() sets it)."""
        if self.TIMESTAMP_FILE is not None:
            timestamp(label=label, timestamp_file=self.TIMESTAMP_FILE)

    def _timed_submit(self, label, fn, retry=True):
        """Run a provider submission call fn(), logged as <label>_submit_start/_end (one pair
        around all attempts, also when it fails, so a failed submission counts as CPU time).
        Returns (fn's result, seconds of all attempts); the attempts are kept in
        self._last_submit_attempts.

        retry=True (real QPUs): an error where the request never reached the provider
        (_submit_error_kind "not_connected") is retried after SUBMIT_RETRY_BACKOFF_S; any other
        error — including a lost answer, where the job may already exist — is not. When it gives
        up it raises ChunkSubmitError (cause, attempts, kind; .submit_s). retry=False (local
        simulator): one call, its error propagates unchanged."""
        self._log_timestamp(f"{label}_submit_start")
        t0       = time.perf_counter()
        attempts = []
        self._last_submit_attempts = attempts
        if not retry:
            out = fn()
            self._log_timestamp(f"{label}_submit_end")
            return out, time.perf_counter() - t0
        for i, delay in enumerate((*SUBMIT_RETRY_BACKOFF_S, None)):
            a0 = time.perf_counter()
            at = now_utc_iso()
            try:
                out = fn()
            except Exception as e:
                kind = _submit_error_kind(e)
                attempts.append({"at": at, "s": round(time.perf_counter() - a0, 3),
                                 "error": scrub_paths(f"{type(e).__name__}: {e}")[:1000], "kind": kind})
                if kind == "not_connected" and delay is not None:
                    print(format_retry(label, "submission", e, i + 2, len(SUBMIT_RETRY_BACKOFF_S) + 1, delay,
                                       noun="next attempt", reason="could not reach the provider"), flush=True)
                    time.sleep(delay)
                    continue
                self._log_timestamp(f"{label}_submit_end")
                err = ChunkSubmitError(label, e, attempts, kind)
                err.submit_s = time.perf_counter() - t0
                raise err from e
            attempts.append({"at": at, "s": round(time.perf_counter() - a0, 3), "error": None, "kind": None})
            self._log_timestamp(f"{label}_submit_end")
            if i:
                print(f"  ✅ [{label}] Submitted on attempt {i + 1}/{len(SUBMIT_RETRY_BACKOFF_S) + 1}.", flush=True)
            return out, time.perf_counter() - t0

    def _submitted_circuit_files(self, nc, n_spam):
        """Which transpiled_circuits/ files a chunk job was built from, in submission order."""
        return (["spam_transpiled.qpy"] if n_spam else []) + [f"chunk_{nc}_transpiled.qpy"]

    def get_results_by_chunk_iqm(self, exp_results_folder_path, nc, adapter, lang=None, backend_name=None):
        """
        Collect results for chunk nc from IQM backend and save to disk.

        :param exp_results_folder_path: str — run folder path (returned by submit())
        :param nc:                      int — chunk index
        :param adapter:                 MQSSQiskitAdapter instance
        :param lang: str — "qiskit" (bitstrings reversed k[::-1]) | "myqlm" (no reversal)
        """
        from mqss_client.job import CircuitJobRequest

        assert lang in ("qiskit", "myqlm"), f"lang must be 'qiskit' or 'myqlm', got {lang!r}"

        exp_results_folder_path = Path(exp_results_folder_path)   # ensure Path object
        record_path        = exp_results_folder_path / "job_records" / f"job_record_chunk_{nc}.json"
        record             = json.loads(record_path.read_text(encoding="utf-8"))
        job_id             = record["job_id"]
        n_spam             = record["n_spam"]
        n_circuits         = record["n_circuits"]
        nshots             = record["nshots"]
        trotter_steps_list = record["trotter_steps_list"]
        roles              = record["roles"]
        nsites             = record["nsites"]
        spam_correction    = record["n_spam"] > 0
        spam_start_idx     = record.get("spam_start", 0)
        circ_start_idx     = record.get("circ_start", n_spam)
        circ_end_idx       = record.get("circ_end",   n_spam + n_circuits) 
        if "backend_name" not in record:
            exit_with_error(f"job record {record.get('job_id')} has no backend_name; upgrade the old records")
        resource_name      = record["backend_name"]
        chunk_key          = f"chunk_{nc}"  

        assert abs((circ_start_idx-spam_start_idx)-n_spam) <1e-8
        assert abs((circ_end_idx-circ_start_idx)-n_circuits) <1e-8

        raw_dir       = exp_results_folder_path / "raw_results"
        counts_file   = raw_dir / "results_counts.json"
        fidelity_file = raw_dir / "results_fidelity.json"

        if record["status"] == "complete" and _chunk_in(counts_file, f"chunk_{nc}") and _chunk_in(fidelity_file, f"chunk_{nc}"):
            print(f"  [Chunk {nc}] Already collected — skipping."); return

        # ── Reconstruct job and retrieve result ───────────────────────────────
        job_type = CircuitJobRequest(
            circuits       = "",
            circuit_format = "qasm",
            resource_name  = resource_name,
            shots          = nshots,
            no_modify      = True,
            queued         = False,
        )
        cache = self.__dict__.setdefault("_iqm_job_cache", {})
        if job_id not in cache:                   # a packed job is shared by several chunks — fetch it once
            print_job_line(f"Chunk {nc}", backend_name or resource_name, MQSS_LRZ, job_id)
            cache.clear()
            cache[job_id] = adapter.client.wait_for_job_result(job_id, job_type)
        result       = cache[job_id]
        # MQSS timestamps are naive Europe/Berlin time — converted to UTC here (source/core/records/job_timing.py).
        # Timing is the whole job's (counted once per job_id by qpu_queue_breakdown).
        eqe_timing   = eqe_collect_timing(result, record.get("submitted_at"),
                                          record.get("n_circuits_in_job", n_spam + n_circuits))
        finished_at  = eqe_timing["execution_end"] or "N/A"
        scheduled_at = eqe_timing["execution_start"] or "N/A"

        spam_raw   = [result.counts[spam_start_idx + si] for si in range(n_spam)]
        all_counts = [result.counts[circ_start_idx + ci] for ci in range(n_circuits)]

        if lang=="qiskit":
            # ── Reverse bitstrings (Qiskit convention) ────────────────────────────
            spam_raw   = [{k[::-1]: v for k, v in c.items()} for c in spam_raw]
            all_counts = [{k[::-1]: v for k, v in c.items()} for c in all_counts]

        # ── Build measurement_counts and fidelity_nec_dict ────────────────────
        measurement_counts = {}
        fidelity_nec_dict  = {}

        for di, nt in enumerate(trotter_steps_list):
            tag = f"Nt_{nt:02d}"
            for ri, role in enumerate(roles):
                # local_idx: ranges 0 to n_depths * len(self.roles) - 1
                # identical across all chunks — same keys appear in every chunk_{nc}
                local_idx = di * len(roles) + ri
                job_key   = f"job_{tag}_{role}_ems{local_idx}"
                circ_key  = f"{tag}_{role}_circuit_0" # always 0 — sample index inside a chunk (1 sample per chunk)
                counts    = all_counts[local_idx]
                if role.endswith("_nec"):
                    fidelity_nec_dict[job_key]  = {circ_key: counts}
                else:
                    measurement_counts[job_key] = {circ_key: counts}

        # ── Save counts and fidelity ──────────────────────────────────────────
        (exp_results_folder_path / "raw_results").mkdir(parents=True, exist_ok=True)
        counts_path   = exp_results_folder_path / "raw_results" / "results_counts.json"
        fidelity_path = exp_results_folder_path / "raw_results" / "results_fidelity.json"

        existing_counts   = load_json(counts_path)   if counts_path.exists()   else {}
        existing_fidelity = load_json(fidelity_path) if fidelity_path.exists() else {}

        existing_counts[chunk_key]   = measurement_counts
        existing_fidelity[chunk_key] = fidelity_nec_dict

        with open(counts_path, "w", encoding="utf-8") as f:
            json.dump(existing_counts, f, indent=4)
        with open(fidelity_path, "w", encoding="utf-8") as f:
            json.dump(existing_fidelity, f, indent=4)

        # ── Save SPAM counts ──────────────────────────────────────────────────
        if spam_correction:
            spam_path     = exp_results_folder_path / "raw_results" / "spam_calibration_counts_per_chunk.json"
            existing_spam = load_json(spam_path) if spam_path.exists() else {}
            spam_counts_per_input = {
                format(si, f"0{nsites}b"): spam_raw[si]
                for si in range(n_spam)
            }
            existing_spam[chunk_key] = spam_counts_per_input
            with open(spam_path, "w", encoding="utf-8") as f:
                json.dump(existing_spam, f, indent=4)

        print_chunk_saved(nc, n_spam, len(measurement_counts), len(fidelity_nec_dict))

        # ── Update job record ─────────────────────────────────────────────────
        record["status"]        = "complete"
        record["scheduled_at"]  = scheduled_at
        record["finished_at"]   = finished_at
        record["result_path"]   = run_relative_path(exp_results_folder_path, "raw_results", "results_counts.json")
        record["circuit_check"] = CC.not_available("IQM EQE1 (MQSS) does not return the executed circuits")
        merge_timing(record, **eqe_timing)
        write_job_record(record_path, record)

    def get_results_by_chunk_iqm_resonance(self, exp_results_folder_path, nc, backend, lang="qiskit", backend_name=None):
        """Retrieve chunk nc from IQM Resonance or VLQ (blocking) and save counts.
        `backend` is an IQMBackend (rebuilt from url+token+quantum_computer), or for a VLQ
        record (service_name 'QAAS_IT4I') the qaas backend (source/core/qpu/iqm_vlq.py)."""
        assert lang in ("qiskit", "myqlm")
        exp_results_folder_path = Path(exp_results_folder_path)
        record_path = exp_results_folder_path / "job_records" / f"job_record_chunk_{nc}.json"
        record      = json.loads(record_path.read_text(encoding="utf-8"))

        raw_dir       = exp_results_folder_path / "raw_results"
        counts_file   = raw_dir / "results_counts.json"
        fidelity_file = raw_dir / "results_fidelity.json"

        if record["status"] == "complete" and _chunk_in(counts_file, f"chunk_{nc}") and _chunk_in(fidelity_file, f"chunk_{nc}"):
            print(f"  [Chunk {nc}] Already collected — skipping."); return

        job_id            = record["job_id"]
        n_spam             = record["n_spam"]
        n_circuits         = record["n_circuits"]
        nshots             = record["nshots"]
        nsites             = record["nsites"]
        trotter_steps_list = record["trotter_steps_list"]
        roles              = record["roles"]
        spam_start_idx     = record.get("spam_start", 0)
        circ_start_idx     = record.get("circ_start", n_spam)
        spam_correction    = n_spam > 0
        chunk_key          = f"chunk_{nc}"

        # ── CHANGE: retrieve via IQMBackend.retrieve_job (no CircuitJobRequest) ──
        vlq   = record.get("service_name") == QAAS_IT4I
        cache = self.__dict__.setdefault("_iqm_job_cache", {})
        if vlq and job_id not in cache:                  # a packed job is shared by several chunks — fetch it once
            print_job_line(f"Chunk {nc}", backend_name or record["backend_name"], QAAS_IT4I, job_id)
            job    = VLQ.retrieve_job(backend, job_id)          # logged, time-limited (source/core/qpu/iqm_vlq.py)
            result = VLQ.wait_for_result(job, job_id)
            if not getattr(result, "success", True):
                record["status"]  = "failed"
                record["failure"] = f"VLQ result not successful: {getattr(result, 'status', None)}"
                record["failure_stage"] = "execution"
                merge_timing(record, **_failed_job_timing(QAAS_IT4I, backend, record))
                write_job_record(record_path, record)
                raise RuntimeError(f"Chunk {nc} VLQ job {job_id} failed: {getattr(result, 'status', None)}")
            all_results = result.get_counts()
            if isinstance(all_results, dict):
                all_results = [all_results]
            cache.clear()
            cache[job_id] = {"all_results": all_results, "job": job,
                             "timing": VLQ.vlq_collect_timing(job, record.get("upload_end_at") or record["submitted_at"])}
        if job_id not in cache:                          # a packed job is shared by several chunks — fetch it once
            print_job_line(f"Chunk {nc}", backend_name or record["backend_name"], IQM_RESONANCE, job_id)
            job = backend.retrieve_job(job_id)
            job.wait_for_final_state()                   # blocks
            if not job.done():
                record["status"]  = "failed"
                record["failure"] = f"job ended {job.status()}"
                record["failure_stage"] = "execution"
                merge_timing(record, **_failed_job_timing(IQM_RESONANCE, backend, record))
                write_job_record(record_path, record)
                raise RuntimeError(f"Chunk {nc} job {job_id} ended {job.status()}: {job.error_message()}")
            all_results = job.result().get_counts()      # list[Counts], one per circuit, submission order
            if isinstance(all_results, dict):            # single-circuit safety
                all_results = [all_results]
            cache.clear()
            cache[job_id] = {"all_results": all_results,
                             # execution window, queue and billed time of the whole job (timeline + raw job)
                             "timing":      iqm_resonance_collect_timing(backend, job, job_id)}
        all_results = cache[job_id]["all_results"]
        n_in_job    = record.get("n_circuits_in_job", circ_start_idx + n_circuits)   # old records: one chunk per job
        assert len(all_results) == n_in_job, \
            f"got {len(all_results)} count dicts, expected {n_in_job}"

        # ── split + reverse bits (identical to get_results_by_chunk_iqm) ──
        spam_raw   = [all_results[spam_start_idx + si] for si in range(n_spam)]
        all_counts = [all_results[circ_start_idx + ci] for ci in range(n_circuits)]
        if lang == "qiskit":
            spam_raw   = [{k[::-1]: v for k, v in c.items()} for c in spam_raw]
            all_counts = [{k[::-1]: v for k, v in c.items()} for c in all_counts]

        # ── Build measurement_counts and fidelity_nec_dict ────────────────────
        measurement_counts = {}
        fidelity_nec_dict  = {}

        for di, nt in enumerate(trotter_steps_list):
            tag = f"Nt_{nt:02d}"
            for ri, role in enumerate(roles):
                # local_idx: ranges 0 to n_depths * len(self.roles) - 1
                # identical across all chunks — same keys appear in every chunk_{nc}
                local_idx = di * len(roles) + ri
                job_key   = f"job_{tag}_{role}_ems{local_idx}"
                circ_key  = f"{tag}_{role}_circuit_0" # always 0 — sample index inside a chunk (1 sample per chunk)
                counts    = all_counts[local_idx]
                if role.endswith("_nec"):
                    fidelity_nec_dict[job_key]  = {circ_key: counts}
                else:
                    measurement_counts[job_key] = {circ_key: counts}

        # ── Save counts and fidelity ──────────────────────────────────────────
        (exp_results_folder_path / "raw_results").mkdir(parents=True, exist_ok=True)
        counts_path   = exp_results_folder_path / "raw_results" / "results_counts.json"
        fidelity_path = exp_results_folder_path / "raw_results" / "results_fidelity.json"

        existing_counts   = load_json(counts_path)   if counts_path.exists()   else {}
        existing_fidelity = load_json(fidelity_path) if fidelity_path.exists() else {}

        existing_counts[chunk_key]   = measurement_counts
        existing_fidelity[chunk_key] = fidelity_nec_dict

        with open(counts_path, "w", encoding="utf-8") as f:
            json.dump(existing_counts, f, indent=4)
        with open(fidelity_path, "w", encoding="utf-8") as f:
            json.dump(existing_fidelity, f, indent=4)

        # ── Save SPAM counts ──────────────────────────────────────────────────
        if spam_correction:
            spam_path     = exp_results_folder_path / "raw_results" / "spam_calibration_counts_per_chunk.json"
            existing_spam = load_json(spam_path) if spam_path.exists() else {}
            spam_counts_per_input = {
                format(si, f"0{nsites}b"): spam_raw[si]
                for si in range(n_spam)
            }
            existing_spam[chunk_key] = spam_counts_per_input
            with open(spam_path, "w", encoding="utf-8") as f:
                json.dump(existing_spam, f, indent=4)

        print_chunk_saved(nc, n_spam, len(measurement_counts), len(fidelity_nec_dict))


        # ── Update job record ─────────────────────────────────────────────────
        # ── execution window, queue and billed time of the job (counted once per job_id by qpu_queue_breakdown) ──
        t = cache[job_id]["timing"]
        if t.get("execution_start") is None or t.get("execution_end") is None:
            print(f"  ⚠️  [Chunk {nc}] No execution window in the job's timeline ({len(t.get('timeline') or [])} events) — timing incomplete.")
        merge_timing(record, **t)
        if vlq:
            check = self._check_executed_iqm_vlq(cache[job_id]["job"], record, exp_results_folder_path, nc)
        else:
            check = self._check_executed_iqm_resonance(backend, record, exp_results_folder_path, nc)
        _warn_circuit_check(f"Chunk {nc}", check)

        record["scheduled_at"]  = t.get("execution_start") or "N/A"
        record["finished_at"]   = t.get("execution_end") or "N/A"
        record["completed_at"]  = t.get("completed_at") or "N/A"
        record["circuit_check"] = check
        record["status"]        = "complete"
        record["result_path"]   = run_relative_path(exp_results_folder_path, "raw_results", "results_counts.json")
        write_job_record(record_path, record)
