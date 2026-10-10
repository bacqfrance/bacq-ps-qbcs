"""
building.py
The circuit-building steps qbcs.build() runs (in local_sim.venv): the BCS Trotter circuits, the
error-mitigation plugin batches, the exact reference observables cached next to the circuits,
and the noise-angle helpers. myQLM is imported only inside the functions that need it, so this module
(like SPAM_functions.py) loads without it; load_exact_observables() reads the cache and computes
any missing depths with myQLM.
"""

import json
import os
import socket
from pathlib import Path

import numpy as np

from source.core.common import get_circuit_folder, rel_path, require_myqlm
on_tgcc = "tgcc" in socket.gethostname() or "irene" in socket.gethostname()

# myQLM is imported inside the functions that use it (circuit building, exact
# observables), so this module loads without it — see source/core/common.py.
def get_default_qpu():
    if on_tgcc:
        from qlmaas.qpus import get_default_qpu as _qlmaas_default_qpu
        return _qlmaas_default_qpu()
    from qat.qpus import PyLinalg
    return PyLinalg()


def rotate_circuit_basis_change(circuit_routine, obs_list_per_qubit=["X", "Y", "Z"]):
    from qat.lang.AQASM import H, I, Program, RX
    nqubits = circuit_routine.arity
    assert len(obs_list_per_qubit)==nqubits, "length of obs_list_per_qubit is not the same as the number of qubits!!"

    basis_rotation_operators = {"X": H, "Y": RX(np.pi/2), "Z": I}

    prog = Program()
    qreg = prog.qalloc(nqubits)
    prog.apply(circuit_routine, qreg)
    
    for qi in range(nqubits):
        obs_key = obs_list_per_qubit[qi]
        prog.apply(basis_rotation_operators[obs_key], qreg[qi])

    circ_rotated = prog.to_circ()
    return circ_rotated


def get_observable_operator(obs_str=None, obs_str_list = ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]):
    from qat.core import Observable, Term
    dict_obs_operators = { 
                          "X" : Observable(3, pauli_terms=[Term(1.0, "X", [0])]), 
                          "Y" : Observable(3, pauli_terms=[Term(1.0, "Y", [1])]), 
                          "Z" : Observable(3, pauli_terms=[Term(1.0, "Z", [2])]), 
                          "XY" : Observable(3, pauli_terms=[Term(1.0, "XY", [0,1])]), 
                          "XZ" : Observable(3, pauli_terms=[Term(1.0, "XZ", [0,2])]), 
                          "YZ" : Observable(3, pauli_terms=[Term(1.0, "YZ", [1,2])]), 
                          "XYZ" : Observable(3, pauli_terms=[Term(1.0, "XYZ", [0,1,2])])
                          }
    obs_ind = obs_str_list.index(obs_str)
    return dict_obs_operators[obs_str], obs_ind

# Building the circuit, and its noisy version
def build_BCS_time_evolution_circuit(nsites, g, epsilon, T, dt, local_qubit_basis=["X", "Y", "Z"], LNN=False):
    from qat.lang.AQASM import Program
    from source.core.circuits.main import BCS_hamiltonian, chain1D
    lattice_1D = chain1D(nsites, bc='OBC')
    Ham = BCS_hamiltonian(g = g, e = epsilon)
    circuit_rout = Ham.build_full_time_evolution_circuit(lattice=lattice_1D, delta_tau=dt, t=T, use_kak=True, LNN=LNN)

    prog = Program()
    q = prog.qalloc(nsites)
    prog.apply(circuit_rout, q)
    circuit_original = prog.to_circ() #myqlm circuit object

    rotated_circuit = rotate_circuit_basis_change(circuit_routine = circuit_rout, obs_list_per_qubit = local_qubit_basis) #myqlm circuit object

    return circuit_original, rotated_circuit # Returns circuit objects

# Define plugins
def build_preplugin(rotated_circuit, config, error_mitigation=None, use_cRC_plugin_for_every_thing=True, compute_classical_nec_map=False, verbose=False):
    from source.core.circuits.main import NEC_plugin, RCCNOT_plugin, cRCCNOT_plugin
    if error_mitigation=="cRC" or error_mitigation=="cRC+NEC" or error_mitigation=="RC" or error_mitigation=="RC+NEC" or error_mitigation=="RAW": 
        only_nec = False
    elif error_mitigation=="NEC":
        only_nec = True    
    else:
        assert 0, "Invalid error mitigation strategy"
    
    depth_label = f"Nt_{config['Nt']:02d}"

    if not only_nec:
        nec = NEC_plugin(circuit_og=rotated_circuit, split_jobs=False,
                         measurement_counts=config["nec_measurement_counts"],
                         compute_classical_nec_map=compute_classical_nec_map,
                         depth_label=depth_label, exp_name=config.get("exp_label", None),        # add
                         folder_name=get_circuit_folder(config, top_folder=True), verbose=verbose)
    else:
        nec = NEC_plugin(circuit_og=rotated_circuit, split_jobs=False, # config["split_raw+nec_jobs"],
                         nsamples=config["samples"], measurement_counts=config["nec_measurement_counts"],
                         compute_classical_nec_map=compute_classical_nec_map,
                         depth_label=depth_label, exp_name=config.get("exp_label", None),        # add
                         folder_name=get_circuit_folder(config, top_folder=True), verbose=verbose)

    if not use_cRC_plugin_for_every_thing:
        rc_plugin = RCCNOT_plugin(
            rng_seed              = config["rng_seed"],
            nsamples              = config["samples"], #1 if only_nec else config["samples"],
            ntrotter              = config["Nt"],
            circuit_og            = rotated_circuit,
            coherent_noise        = True,
            coherent_noise_angles = config["noise_angles"],
            log_batch             = True,
            log_twirls            = not only_nec,
            exp_name              = config.get("exp_label", None), #config["exp_label_nec"] if only_nec else config["exp_label"],
            turn_off_RC           = only_nec,
            batch_label           = error_mitigation,
            folder_name           = get_circuit_folder(config, top_folder=False),
            randomize_nec_input_state = config["nec_randomize_input_state"],
            LNN                   = config["lnn_architecture"]
        )
        return nec, rc_plugin
    else:
        rc_plugin = cRCCNOT_plugin(
            rng_seed                  = config["rng_seed"],
            nsamples                  = config["samples"], #1 if only_nec else config["samples"],
            ntrotter                  = config["Nt"],
            circuit_og                = rotated_circuit,
            coherent_noise            = True,
            coherent_noise_angles     = config["noise_angles"],
            log_batch                 = True,
            log_twirls                = not only_nec,
            exp_name                  = config.get("exp_label", None), #config["exp_label_nec"] if only_nec else config["exp_label"],
            turn_off_RC               = only_nec,
            batch_label               = error_mitigation,
            neighbor_noise            = True,
            neighbor_noise_angles     = config["neighbor_noise_angles"],
            turn_off_cRC              = (error_mitigation!="cRC+NEC"),
            turn_off_error_mitigation = only_nec,
            folder_name               = get_circuit_folder(config, top_folder=False),
            randomize_nec_input_state = config["nec_randomize_input_state"],
            LNN                       = config["lnn_architecture"],
            verbose                   = verbose
        )
        # No compilation is done inside this function
        return nec, rc_plugin


def build_postplugin(config, obs_list, shots_list, error_mitigation=None, job_tags=None, rc_samples_per_chunk=None, chunk_index=None, ndepths=1, ems_list=None, infidelity_from_counts=None):
    from source.core.circuits.main import RC_NEC_post_compile
    if error_mitigation=="RC" or error_mitigation=="RC+NEC" or error_mitigation=="RAW": 
        only_nec = False
    elif error_mitigation=="NEC":
        only_nec = True    
    elif error_mitigation=="all_split":
        only_nec = False
    else:
        assert 0, "Invalid error mitigation strategy"    
    if rc_samples_per_chunk is None:
        rc_samples_per_chunk = config["samples"]
    
    assert ems_list, "ems_list must be provided"
    assert infidelity_from_counts is not None, "infidelity_from_counts must be explicitly True or False"

    rc_nec_pp = RC_NEC_post_compile(
        obs_list      = obs_list,
        shots_list    = shots_list,
        exp_name      = config.get("exp_label_nec", None) if only_nec else config.get("exp_label", None),
        nqubits       = config["nsites"],
        rng_seed      = config["rng_seed"],
        job_tags      = job_tags,
        chunk_samples_per_mitigation_strategy = 1 if only_nec else rc_samples_per_chunk,
        samples_per_mitigation_strategy       = 1 if only_nec else config["samples"],
        chunk_index   = chunk_index,
        folder_name   = config["folder_path"],
        ndepths       = ndepths,
        nshots        = config["shots"],
        ems_list      = ems_list,
        infidelity_from_counts = infidelity_from_counts
    )
    return rc_nec_pp

# Define pipeline
# Initial batch
def prepare_initial_batch(circuit, shots, obs=None, job_type=None):
    # --------------- Preparing the initial batch of jobs -----------------------
    from qat.core import Batch
    job                   = circuit.to_job(nbshots=shots, observable=obs)
    job.meta_data         = {}
    job.meta_data["role"] = "org" if job_type is None else job_type
    job.meta_data["gid"]  = "job0"
    init_batch_jobs       = Batch([job])

    return init_batch_jobs

# compile batch with the relevant plugin or the load the batch file
def load_or_compile_batch(plugin_for_compile, batch_jobs, filepath, verbose=False):
    from qat.core import Batch
    if filepath is None or not os.path.exists(filepath):
        return plugin_for_compile.compile(batch_jobs, None)
    batch = Batch.load(filepath)
    if verbose: print(f"Loaded {len(batch.jobs)} jobs from {rel_path(filepath)}")
    return batch


def load_obs_and_shots(config, error_mitigation=None):
    n_jobs = 2
    obs_list   = [None] * n_jobs
    shots_list = [config["shots"]] * n_jobs
    return obs_list, shots_list
    
def compute_exact_observables(circuit_original, return_jobs=False):
    from qat.core import Batch
    qpu = get_default_qpu()
    exact_values = np.zeros(7, dtype=complex)
    obs_str_list = ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]

    job_list = [None]*7
    for obs_str in obs_str_list:
        obs, obs_ind = get_observable_operator(obs_str, obs_str_list=obs_str_list)
        job_exact_obs = circuit_original.to_job(observable=obs)
        job_list[obs_ind] = job_exact_obs
        if on_tgcc:
            async_res = qpu.submit(job_exact_obs)
            async_res.join()
            exact_values[obs_ind] = async_res.value
        else:    
            exact_values[obs_ind] = qpu.submit(job_exact_obs).value
    if not return_jobs:
        return exact_values
    else:
        return exact_values, Batch(job_list), obs_str_list


# ── Cache of exact (noise-free) observables ───────────────────────────────────
# The analysis compares every measured result against the noise-free values of
# the 7 observables at each depth. They depend only on the Hamiltonian and the
# depth, so build() computes them once (myQLM, local_sim.venv) and saves them
# next to the circuits; the analysis then only reads a JSON file.
EXACT_OBS_LIST      = ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]   # order of compute_exact_observables()
EXACT_OBS_FILENAME  = "exact_observables.json"

def exact_observables_path(CONFIG):
    return Path(get_circuit_folder(CONFIG, top_folder=True)) / EXACT_OBS_FILENAME

def _exact_observables_key(CONFIG, use_circuit_LNN):
    """Everything the exact values depend on, besides the depth."""
    return json.dumps({
        "nsites":            int(CONFIG["nsites"]),
        "g":                 float(CONFIG["g"]),
        "epsilon":           np.asarray(CONFIG["epsilon"]).tolist(),
        "dt":                float(CONFIG["dt"]),
        "local_qubit_basis": list(CONFIG["local_qubit_basis"]),
        "LNN":               bool(use_circuit_LNN),
    }, sort_keys=True)

def _compute_exact_observables_myqlm(CONFIG, nt, use_circuit_LNN):
    circuit_kwargs = {k: CONFIG[k] for k in ["nsites", "g", "epsilon", "dt", "local_qubit_basis"]}
    circuit_kwargs["T"]   = nt * CONFIG["dt"]
    circuit_kwargs["LNN"] = use_circuit_LNN
    circuit_original, _ = build_BCS_time_evolution_circuit(**circuit_kwargs)
    return compute_exact_observables(circuit_original=circuit_original)

def missing_exact_observables(CONFIG, trotter_steps_list, use_circuit_LNN):
    """Depths (ints) whose exact observables aren't cached yet."""
    path   = exact_observables_path(CONFIG)
    entry  = json.loads(path.read_text(encoding="utf-8")).get(_exact_observables_key(CONFIG, use_circuit_LNN)) if path.exists() else None
    cached = {} if entry is None else entry["values"]
    return [int(nt) for nt in trotter_steps_list if str(int(nt)) not in cached]

def cache_exact_observables(CONFIG, trotter_steps_list, use_circuit_LNN):
    """Compute (myQLM) and save the exact observables of every depth not cached yet.
    Values are stored as [real, imag] pairs, so reading them back gives exactly
    what compute_exact_observables() returned."""
    path  = exact_observables_path(CONFIG)
    data  = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key   = _exact_observables_key(CONFIG, use_circuit_LNN)
    entry = data.setdefault(key, {"observables": EXACT_OBS_LIST, "values": {}})
    missing = [int(nt) for nt in trotter_steps_list if str(int(nt)) not in entry["values"]]
    if not missing:
        return
    require_myqlm(f"Computing the exact observables for Nt={missing}")
    for nt in missing:
        values = _compute_exact_observables_myqlm(CONFIG, nt, use_circuit_LNN)
        entry["values"][str(nt)] = [[float(v.real), float(v.imag)] for v in values]
    entry["values"] = dict(sorted(entry["values"].items(), key=lambda kv: int(kv[0])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  [Exact] Cached exact observables for Nt={missing} → {rel_path(path)}")

def load_exact_observables(CONFIG, trotter_steps_list, use_circuit_LNN):
    """{nt: exact observables (complex array, EXACT_OBS_LIST order)} for the given
    depths, from the cache written by build(). Depths missing from the cache are
    computed with myQLM and cached first."""
    path    = exact_observables_path(CONFIG)
    missing = missing_exact_observables(CONFIG, trotter_steps_list, use_circuit_LNN)
    if missing:
        print(f"  ⚠️  [Exact] Nt={missing} not cached in {rel_path(path)} — computing with myQLM. "
              f"Run build() in local_sim.venv to cache them.")
        cache_exact_observables(CONFIG, missing, use_circuit_LNN)
    values = json.loads(path.read_text(encoding="utf-8"))[_exact_observables_key(CONFIG, use_circuit_LNN)]["values"]
    return {int(nt): np.array([complex(re, im) for re, im in values[str(int(nt))]]) for nt in trotter_steps_list}


def calculate_infidelity_theta(theta_1, theta_2, theta_3, noise="dep"):
    if noise=="dep":
        I_dir = theta_1**2/2 + theta_2**2/2
        I_ct  = theta_3**2/2
        I_tot = I_dir + I_ct
    elif noise=="pauli":
        I_dir = 3*theta_1**2/16 + 3*theta_2**2/16
        I_ct  = 3*theta_3**2/16
        I_tot = I_dir + I_ct
    else:
        raise ValueError(f"Unknown noise model: '{noise}'. Choose 'dep' or 'pauli'.")
    return I_tot

def calculate_thetas_from_infidelity(I_tot, r_ct, noise="dep"):
    """
    Given I_tot and r_ct = I_ct / I_dir, returns theta_1 (=theta_2) and theta_3.
    
    From the forward function:
      dep:   I_dir = theta_1^2/2 + theta_2^2/2 = theta_1^2  (since theta_1=theta_2)
             I_ct  = theta_3^2/2
      pauli: I_dir = 3*theta_1^2/16 + 3*theta_2^2/16 = 3*theta_1^2/8
             I_ct  = 3*theta_3^2/16
    """
    # I_ct / I_dir = r_ct  and  I_ct + I_dir = I_tot
    # => I_dir = I_tot / (1 + r_ct)
    # => I_ct  = I_tot * r_ct / (1 + r_ct)
    I_dir = I_tot / (1 + r_ct)
    I_ct  = I_tot * r_ct / (1 + r_ct)

    if noise == "dep":
        # I_dir = theta_1^2  => theta_1 = sqrt(I_dir)
        theta_1 = np.sqrt(I_dir)
        # I_ct = theta_3^2/2 => theta_3 = sqrt(2 * I_ct)
        theta_3 = np.sqrt(2 * I_ct)

    elif noise == "pauli":
        # I_dir = 3*theta_1^2/8 => theta_1 = sqrt(8*I_dir/3)
        theta_1 = np.sqrt(8 * I_dir / 3)
        # I_ct = 3*theta_3^2/16 => theta_3 = sqrt(16*I_ct/3)
        theta_3 = np.sqrt(16 * I_ct / 3)

    else:
        raise ValueError(f"Unknown noise model: '{noise}'. Choose 'dep' or 'pauli'.")

    theta_2 = theta_1  # by assumption
    return theta_1, theta_2, theta_3
