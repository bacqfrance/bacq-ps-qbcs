import numpy as np
from qat.fermion import SpinHamiltonian
from qat.core import Term, Observable
from qat.core import Batch, BatchResult, Job
from qat.lang.AQASM import H, Z, I, X, Y, RX, RY, RZ, CNOT, QRoutine, Program
from source.core.circuits.circuit_functions import * # works locally and on cluster 
# from mpi4py import MPI 
import scipy
from qat.core.wrappers.result import Result
from source.core.circuits.rc_functions import *
import json 
import copy 
from pathlib import Path
from qat.core.plugins import AbstractPlugin
from tqdm import tqdm
from copy import deepcopy
import gc
from qat.qpus import PyLinalg
import os
# myQLM-free helpers (paths, NEC target bitstrings) live in source/core/common.py —
# re-exported here so `from source.core.circuits.main import ...` keeps working.
from source.core.common import *


sqrt_X = QRoutine()
q = sqrt_X.new_wires(1)
sqrt_X.apply(RZ(-np.pi/2), q[0])
sqrt_X.apply(H,            q[0])
sqrt_X.apply(RZ(-np.pi/2), q[0])


class chain1D:
    def __init__(self, l, bc="PBC"):
        """
        Initializes a 1D lattice.

        Parameters:
        - l: number of lattice sites.
        - bc: boundary condition
        """
        self.nd = (int(l), )
        self.n = int(l)
        l = int(l)
        # Use zip to pair x and y coordinates
        x = np.tile(np.arange(l), 1)
        y = np.repeat(np.arange(1), l)
        self.coor = np.array(list(zip(x, y)))
        self.indices = np.arange(self.n)
        self.bc=bc

def make_swap(q_routine, reg, i, j):
    q_routine.apply(CNOT, reg[i], reg[j])
    q_routine.apply(CNOT, reg[j], reg[i])
    q_routine.apply(CNOT, reg[i], reg[j])

class BCS_hamiltonian:
    def __init__(self, g, e):
        self.g = g # scalar
        self.e = e # np array
        self.gstate = None

    def build_hamiltonian(self, lattice):
        nsites = lattice.n
        assert len(self.e)==nsites, "No. of lattice sites is inconsistent"
        nqubits = nsites # Hard-core boson mapping - NOT JW mapping!
        
        hamiltonian_terms = []
        for i in range(nsites):
            hamiltonian_terms.append(Term((0.5*self.g - self.e[i]), "Z", [i])) # On-site terms
            for j in range(i+1, nsites):
                hamiltonian_terms.append(Term(-0.5*self.g, "XX", [i, j])) # Coupling terms
                hamiltonian_terms.append(Term(-0.5*self.g, "YY", [i, j])) # Coupling terms            

        Ham = SpinHamiltonian(nqubits, hamiltonian_terms)
        return Ham
    
    def build_trotter_step(self, lattice, delta_tau):
        nsites = lattice.n
        q_routine = QRoutine()
        nqubits = nsites
        q_reg = q_routine.new_wires(nqubits)

        for i in range(nsites):
            q_routine.apply(RZ((self.g - 2*self.e[i])*delta_tau), q_reg[i]) 

        for i,j in [(0,1), (1,2), (0,2)]:
            XX_gate(q_routine, q_reg, -self.g*delta_tau, i, j)
            YY_gate(q_routine, q_reg, -self.g*delta_tau, i, j)
            
        return q_routine 
    
    def build_trotter_step_KAK(self, lattice, delta_tau, parametrized_twirling=False, twirling_key=None, coherent_noise=None):
        nsites = lattice.n
        q_routine = QRoutine()
        nqubits = nsites
        q_reg = q_routine.new_wires(nqubits)

        if parametrized_twirling:
            assert len(twirling_key) == int(nsites*(nsites-1)), "twirling_key should be a list of str!"

        count_index = 0

        for i in range(nsites):
            q_routine.apply(RZ((self.g - 2*self.e[i])*delta_tau), q_reg[i])

        for i,j in [(0,1), (1,2), (0,2)]:
            if not parametrized_twirling: 
                rot_XY(q_routine, q_reg, -self.g*delta_tau, i, j)
            else:
                rot_XY_RC(q_routine, q_reg, -self.g*delta_tau, i, j, twirling_key[count_index:count_index+2], coherent_noise=coherent_noise)
            count_index += 2

        return q_routine  


    def build_trotter_step_KAK_LNN(self, lattice, delta_tau, qubit_label):
        """
        LNN-compliant Trotter step using 1 SWAP only.

        qubit_label : list of 3 ints — logical qubit at each physical position.
               qubit_label[p] = q means logical qubit q is currently at physical wire p.

               Step A (normal order):    qubit_label = [0, 1, 2]  → ends as [0, 2, 1]
               Step B (permuted order):  qubit_label = [0, 2, 1]  → ends as [0, 1, 2]

        Returns
        -------
        q_routine  : QRoutine — the Trotter step circuit
        qubit_label : list of 3 ints — logical qubit order after this step
        """
        qrout            = QRoutine()
        physical_reg     = qrout.new_wires(lattice.n)
        theta            = -self.g * delta_tau
        nsites           = lattice.n

        # RZ — use perm to pick correct epsilon for each physical wire
        for p in range(nsites):
            qrout.apply(RZ((self.g - 2*self.e[qubit_label[p]])*delta_tau), physical_reg[p])

        # Nearest-neighbour rot_XY on (p0,p1) and (p1,p2) — always LNN
        rot_XY(qrout, physical_reg, theta, 0, 1)   # (perm[0], perm[1])
        rot_XY(qrout, physical_reg, theta, 1, 2)   # (perm[1], perm[2])

        # SWAP(p1,p2) to bring perm[2] adjacent to p0 for the (0,2) interaction
        make_swap(qrout, physical_reg, 1, 2)
        qubit_label = [qubit_label[0], qubit_label[2], qubit_label[1]]   # track permutation after SWAP

        # Now perm[2] is at p1 — rot_XY(p0,p1) implements (perm[0], perm[2]) ✓ LNN
        rot_XY(qrout, physical_reg, theta, 0, 1)

        return qrout, qubit_label

    
    def build_full_time_evolution_circuit(self, lattice, delta_tau, t, use_kak=True, use_circuit_from_paper=False, return_circuit=False, LNN=False):
        
        def initialization_block(rout, reg):
            """
            Initializes the state as per the article Phys. Rev. R 6, 013142
            """
            assert lattice.n == 3, "This initialization only works for 3 qubits!"
            rout.apply(RZ(-np.pi), reg[0])
            rout.apply(sqrt_X,   reg[0])
            rout.apply(RZ(0.43),   reg[0])
            rout.apply(sqrt_X,   reg[0])

            rout.apply(RZ(-np.pi/2), reg[1])
            rout.apply(sqrt_X,     reg[1])
            rout.apply(RZ(np.pi/2),  reg[1])

            rout.apply(RZ(-np.pi), reg[2])
            rout.apply(sqrt_X,   reg[2])
            rout.apply(RZ(2.71),   reg[2])
            rout.apply(sqrt_X,   reg[2])
        
        n_steps = int(round(t / delta_tau))
        if abs(t - n_steps * delta_tau) > 1e-12:
            raise ValueError(f"Choose t as multiple of dt (got t={t}, dt={delta_tau})")

        nsites = lattice.n
        nqubits = nsites
        circ_rout = QRoutine()
        circ_reg = circ_rout.new_wires(nqubits)
        initialization_block(circ_rout, circ_reg)

        if n_steps:
            if LNN:
                # Alternating A/B steps — 1 SWAP per step, qubit_label tracks
                # which logical qubit sits at each physical wire at all times
                qubit_label = [0, 1, 2]
                for _ in range(n_steps):
                    step, qubit_label = self.build_trotter_step_KAK_LNN(
                        lattice, delta_tau, qubit_label
                    )
                    circ_rout.apply(step, circ_reg)
                # If odd n_steps, qubit_label ends as [0,2,1] — restore with SWAP
                if n_steps%2 == 1:
                    assert qubit_label == [0, 2, 1], \
                        f"Unexpected final qubit_label={qubit_label} — check LNN step logic"
                    make_swap(circ_rout, circ_reg, 1, 2)

            elif not use_circuit_from_paper:
                step = (self.build_trotter_step_KAK if use_kak else self.build_trotter_step)(lattice, delta_tau=delta_tau)
            else:
                assert delta_tau==0.2, "The circuit in the paper is only adapted for dt=0.2!!!"
                step = build_trotter_step_paper_IBM()
        
            if not LNN:    
                for _ in range(n_steps):
                    circ_rout.apply(step, circ_reg)
        
        if not return_circuit:
            return circ_rout
        else:
            prog = Program()
            qreg = prog.qalloc(nqubits)
            prog.apply(circ_rout, qreg)
            return prog.to_circ()



NAMED_1Q = {"H": H, "X": X, "Y": Y, "Z": Z, "I": I}
PARAM_1Q = {"RZ": RZ, "RX": RX, "RY": RY}

def unpack(item):
    # Op-like
    if hasattr(item, "gate"):
        g, qbits = item.gate, item.qbits
        params = getattr(item, "params", None)
        return g, qbits, params
    # tuple-like (gate, qbits, params?) depending on version
    g, qbits = item[0], item[2]
    params = item[1] if len(item) > 2 else None
    return g, qbits, params


def prep_computational_basis_input(prog, q, input_bitstring):
    for qi, bit in enumerate(input_bitstring):
        if bit == "1":
            prog.apply(X, q[qi])

def build_projector_observable(target_bitstring, nqubits):
    target_index = int(target_bitstring, 2)
    v = np.zeros(2**nqubits, dtype=complex)
    v[target_index] = 1
    M = np.outer(v, v.conj())
    return Observable(nqubits, matrix=M)

def classically_simulate_skeleton_qlm(skeleton_circuit, nqubits, qpu=None):
    """
    Runs the skeleton circuit through an exact myQLM simulator for every
    computational-basis input, and returns {input_bitstring: output_bitstring}.
    Works for any entangling gate (CNOT, CZ, SWAP, ...) that maps
    computational-basis states to computational-basis states (up to phase) —
    no per-gate-type logic needed.
    """
    qpu = qpu if qpu is not None else PyLinalg()
    outputs = {}

    for state in range(2**nqubits):
        input_bitstring = format(state, f"0{nqubits}b")

        prog = Program()
        q = prog.qalloc(nqubits)
        for qi, bit in enumerate(input_bitstring):
            if bit == "1":
                prog.apply(X, q[qi])

        # Re-apply the skeleton's own gates on top of the prepared input
        for item in skeleton_circuit.iterate_simple():
            gate, qbits, params = unpack(item)
            name = getattr(gate, "name", None)
            gname = name if name is not None else gate

            if gname in ["CNOT", "dCNOT"]:
                prog.apply(CNOT, *[q[i] for i in qbits])          # ideal action, regardless of noisy variant
            elif hasattr(gate, "arity"):
                prog.apply(gate, *[q[i] for i in qbits])
            else:
                raise TypeError(f"Unexpected gate type in skeleton: {gate!r}")

        job = prog.to_circ().to_job(nbshots=0)   # nbshots=0 → exact amplitudes
        result = qpu.submit(job)

        best = max(result, key=lambda s: s.probability)
        assert best.probability > 1 - 1e-9, (
            f"Skeleton produced a superposition for input {input_bitstring} "
            f"(max amplitude probability {best.probability:.6f}) — the skeleton "
            f"contains a gate that doesn't preserve computational-basis states."
        )
        output_state = int.from_bytes(best._state, "big")
        outputs[input_bitstring] = format(output_state, f"0{nqubits}b")

    return outputs


class RCCNOT_plugin(AbstractPlugin):
    """
    Randomized-compiling plugin for circuits containing CNOT-like gates.

    This plugin replaces each CNOT or dCNOT in a circuit with a twirled
    rcCNOT implementation, generates multiple randomized circuit samples,
    and averages the execution results over those samples.

    Main features:
    - Pauli twirling of 2-qubit gates
    - support for ideal CNOT and noisy dCNOT circuits
    - optional exact twirling
    - optional logging of twirling patterns, compiled batches, and results

    The plugin expands each input job into `nsamples` randomized jobs during
    `compile()`, and reduces them back to one averaged result per original job
    during `post_process()`.
    """
    def __init__(self, rng_seed=0, nsamples=4, ntrotter=None, circuit_og=None,
                 coherent_noise=False, coherent_noise_angles=None, turn_off_RC=False,
                 same_samples_rc_nec=True, log_batch=False, log_twirls=False,
                 exp_name=None, exact_twirling=False, batch_label=None,
                 folder_name="datafiles",
                 randomize_nec_input_state=True,
                 run_nec_obs_job=False, LNN=False, verbose=False):
        super().__init__()
        self.verbose = verbose
        assert ntrotter is not None 
        assert circuit_og is not None
        assert batch_label is not None
        
        if coherent_noise: assert coherent_noise_angles is not None, "coherent_noise_angles must be a list[float] of length 6."

        self.rng_seed = rng_seed
        self.nsamples = int(nsamples)
        self.trotter_steps = int(ntrotter)
        self.nqubits = circuit_og.nbqbits
        self.same_samples_rc_nec = same_samples_rc_nec
        self.log_data = log_batch
        self.log_twirls = log_twirls
        self.exp_name = exp_name
        self.batch_label = batch_label
        self.folder_name = folder_name
        self.samples_for_naming = int(nsamples)
        self.turn_off_RC = turn_off_RC
        self.randomize_nec_input_state = randomize_nec_input_state
        self.run_nec_obs_job = run_nec_obs_job
        self.LNN = LNN

        if not turn_off_RC:
            self.Pauli2_keys = [f"{i}{j}" for i in range(4) for j in range(4)] # ["00", "01", "02", "03", "10", "11", "12", "13", "20", "21", "22", "23", "30", "31", "32", "33"]: "ij" -> P_i x P_j on the right side of CNOT; P_0=I, P_1=X, P_2=Y, P_3=Z
            if verbose: print("Building jobs for RC+NEC")
        else:
            self.Pauli2_keys = ["00"]*16
            if verbose: print("Building jobs for NEC")

        if self.LNN:
            self.n_CNOTs = int((scipy.special.comb(self.nqubits, 2)*2 + 3) * self.trotter_steps + int(self.trotter_steps % 2) * 3)
        else:
            self.n_CNOTs = int(scipy.special.comb(self.nqubits, 2)*2*self.trotter_steps)

        self.cohr_noise = coherent_noise
        self.cohr_noise_angles = coherent_noise_angles
        self.exact_twirling = exact_twirling
        if self.exact_twirling: self.nsamples = 16**(self.n_CNOTs)
        assert self.nsamples<2e3, "Extremely large sample size"
        self.measurement_counts = {}

    def twirl_CNOTs(self, circuit, index, role=None):
        '''
        Replaces the CNOTS by custom rcCNOT gates in the circuit
        
        :param circuit: Circuit to replace CNOTS
        :param index: a tuple containing indices of job type, and samples

        Returns a circuit object
        '''
        prog = Program()
        q = prog.qalloc(circuit.nbqbits)

        assert role is not None, "role cannot be None!!"
        if role.endswith("_nec") and self.nsamples > 1 and self.randomize_nec_input_state:
            input_bitstring = sample_index_to_input_bitstring(index[1], self.nsamples, self.nqubits)
            prep_computational_basis_input(prog, q, input_bitstring)

        assert index[1]<self.nsamples, "The sample_index is invalid!!"

        list_indices = self.twirl_indices_right[index[0], index[1], :] 
        assert len(list_indices) == self.n_CNOTs, "The length of twirling indices should be the same as the number of CNOTs!"
        count_CNOTs = 0
        count_2q_out = 0
        for item in circuit.iterate_simple():
            gate, qbits, params = unpack(item)
            name = getattr(gate, "name", None)
            gname = name if name is not None else gate

            if gname == "CNOT":
                assert len(qbits)==2, "CNOT must act on 2 qubits"
                key = self.Pauli2_keys[list_indices[count_CNOTs]]

                prog.apply(rcCNOT(key, coherent_noise=False, angles=None), q[qbits[0]], q[qbits[1]])
                count_CNOTs += 1
                count_2q_out += 1
                continue
            
            # if gate already object other than CNOT, apply
            elif hasattr(gate, "arity"):
                prog.apply(gate, *[q[i] for i in qbits])
                if not isinstance(gate, str) and hasattr(gate, "arity") and gate.arity == 2:
                    count_2q_out += 1
                continue
            
            # gate is a string name: build the gate object
            elif not isinstance(gate, str):
                raise TypeError(f"Unexpected gate type: {type(gate)}")

            if gate in PARAM_1Q:
                theta = params[0] if isinstance(params, (list, tuple)) else params
                prog.apply(PARAM_1Q[gate](theta), q[qbits[0]])
            elif gate in NAMED_1Q:
                prog.apply(NAMED_1Q[gate], q[qbits[0]])
            else:
                raise RuntimeError(f"Unsupported string gate {gate!r} (need a resolver for it).")
        
        assert count_CNOTs==self.n_CNOTs, "No. of CNOTS is not consistent!!"
        assert count_2q_out==self.n_CNOTs, "No. of CNOTS is not consistent!!"
        twirled_circuit = prog.to_circ()
        return twirled_circuit

    def twirl_dCNOTs(self, circuit, index, role=None):
        '''
        Replaces the CNOTS or noisy_CNOTs by custom rcCNOT gates in the circuit
        
        :param circuit: Circuit to replace CNOTS
        :param index: a tuple containing indices of job type, and samples

        Returns a circuit object
        '''
        prog = Program()
        q = prog.qalloc(circuit.nbqbits)

        assert role is not None, "role cannot be None!!"
        if role.endswith("_nec") and self.nsamples > 1 and self.randomize_nec_input_state:
            input_bitstring = sample_index_to_input_bitstring(index[1], self.nsamples, self.nqubits)
            prep_computational_basis_input(prog, q, input_bitstring)

        assert index[1]<self.nsamples, "The sample_index is invalid!!"
        list_indices = self.twirl_indices_right[index[0], index[1], :] 
        count_CNOTs = 0
        count_2q_out = 0
        for op in circuit.ops:
            gdef = circuit.gateDic.get(op.gate) if isinstance(op.gate, str) else None
            if gdef:
                gname = gdef.syntax.name
                syntax_params = gdef.syntax.parameters
                arity = gdef.arity
            qbits = op.qbits

            param_list = []
            for p in syntax_params or []:
                if getattr(p, "double_p", None) is not None:
                    param_list.append(float(p.double_p))
                elif getattr(p, "int_p", None) is not None:
                    param_list.append(float(p.int_p))
                else:
                    param_list.append(None)

            if gname in ["dCNOT", "CNOT"]:
                assert arity==2, "CNOT must act on 2 qubits"
                key = self.Pauli2_keys[list_indices[count_CNOTs]]
                if gname=="dCNOT":
                    assert np.linalg.norm(np.array(self.cohr_noise_angles) - np.array(param_list)) < 1e-16, "Angles of dCNOT are not consistent!!"
                prog.apply(rcCNOT(key, coherent_noise=self.cohr_noise, angles=self.cohr_noise_angles), q[qbits[0]], q[qbits[1]])
                count_CNOTs += 1
                count_2q_out += 1
                continue
            elif arity==2:
                raise RuntimeError(f"Don't know how to apply gate {gname!r} (id={op.gate!r})")
            
            if gname in PARAM_1Q:
                theta = param_list[0] if isinstance(param_list, (list, tuple)) else param_list
                prog.apply(PARAM_1Q[gname](theta), q[qbits[0]])
            elif gname in NAMED_1Q: 
                prog.apply(NAMED_1Q[gname], q[qbits[0]])
            else:
                raise RuntimeError(f"Unsupported string gate {gname!r} (need a resolver for it).")
        
        assert count_CNOTs==self.n_CNOTs, "No. of CNOTS is not consistent!!"
        assert count_2q_out==self.n_CNOTs, "No. of CNOTS is not consistent!!"
        twirled_circuit = prog.to_circ()
        return twirled_circuit


    def compile(self, batch, specs=None):
        '''
        Produces a new batch of compiled circuits from a given batch of circuits.
        
        :param batch: A batch of circuits.

        Returns a new batch of compiled circuits - len() of the final batch.jobs = len() of initial batch.jobs * nsamples
        '''
        njobs = len(batch.jobs)
        rng = np.random.default_rng(self.rng_seed)

        if self.same_samples_rc_nec:
            base = rng.choice(16, size=(self.nsamples, self.n_CNOTs))
            if self.exact_twirling: 
                twirl_opt = [16]*self.n_CNOTs 
                base = np.indices(tuple(twirl_opt)).reshape(self.n_CNOTs,-1).T
                assert base.shape[0]==self.nsamples
            self.twirl_indices_right = np.repeat(base[None, :, :], njobs, axis=0)
        else:
            self.twirl_indices_right = rng.choice(16, size=(njobs, self.nsamples, self.n_CNOTs))
            assert not self.exact_twirling, "Exact twirling not allowed here!"

        self.obs_list = [] #batch.jobs[0].observable
        self.shots_list = []
        # assert self.obs is not None, "only observables!!"
        new_RC_jobs = []
        for ji, job in enumerate(batch.jobs):
            if not self.turn_off_RC:
                self.obs_list.append(job.observable)
                self.shots_list.append(job.nbshots)

            # assert job.observable is not None
            assert job.meta_data is not None, "meta_data of the submitted job should not be empty!!"
            job_role = job.meta_data["role"]
            for si in range(self.nsamples):
                if not self.cohr_noise: new_circuit = self.twirl_CNOTs(job.circuit,  index=(ji,si), role=job_role)
                if self.cohr_noise: new_circuit     = self.twirl_dCNOTs(job.circuit, index=(ji,si), role=job_role)

                if self.run_nec_obs_job and job_role.endswith("_nec") and self.nsamples > 1 and self.randomize_nec_input_state and job.observable is not None:
                    input_bitstring  = sample_index_to_input_bitstring(si, self.nsamples, self.nqubits)
                    target_bitstring = job.meta_data["classical_nec_map"][input_bitstring]
                    obs_for_job = build_projector_observable(target_bitstring, self.nqubits)
                else:
                    obs_for_job = job.observable

                new_job = new_circuit.to_job(observable=obs_for_job, nbshots=job.nbshots)
                new_job.meta_data = {}
                new_job.meta_data["role"] = job.meta_data["role"]
                new_job.meta_data["gid"] = job.meta_data["gid"] + f"_rcc{si}"
                new_RC_jobs.append(new_job)
        # ---------------------- Correction of obs_list and shots_list ------------------------
        if self.turn_off_RC and njobs>1:
            self.obs_list   = [batch.jobs[0].observable, batch.jobs[njobs//2].observable] 
            self.shots_list = [batch.jobs[0].nbshots,    batch.jobs[njobs//2].nbshots]
        if self.turn_off_RC and njobs==1:
            self.obs_list   = [batch.jobs[0].observable] 
            self.shots_list = [batch.jobs[0].nbshots]

        # Return compiled batch
        batch_jobs = Batch(jobs=new_RC_jobs)

        if self.log_data or self.log_twirls: self.save_data(batch_jobs, save_batch=self.log_data, save_twirls=self.log_twirls, exp_label=self.exp_name)
        return batch_jobs

    def post_process(self, batch_result):
        '''
        Average results over randomized compiled circuit samples, for each job type - org, nec, ....

        :param batch_result: BatchResult object. BatchResult.results is the list of all results (nsamples*njobs)

        Returns a new BatchResult object, with len(BatchResult.results) = njobs 
        '''
        grouped_results = [
            batch_result.results[i:i+self.nsamples]
            for i in range(0, len(batch_result.results), self.nsamples)
        ]    # A List of lists
        
        assert len(grouped_results)==len(self.obs_list), "Consistency check regarding the shape of grouped results"

        pp_results = []
        for ind_group,results in enumerate(grouped_results):
            avg_value = 0
            if self.obs_list[ind_group] is not None:
                for result in results:
                    avg_value+=result.value
                avg_value /= len(results)
                pp_results.append(Result(value=avg_value))                
            else:
                # assert self.obs is not None, "For now, the observable must be defined!!"
                shots = self.shots_list[ind_group]
                if shots == 0: shots = 1e14
                dict_counts = {} # Each group of RC jobs (with a given job id) - for a given circuit, has the same dictionary
                
                for ri,result in enumerate(results): # iterate over RC sample circuit results
                    counts = {}
                    for s in result:
                        state = int.from_bytes(s._state, "big")
                        bitstring = format(state, f"0{self.nqubits}b")
                        counts[bitstring] = counts.get(bitstring, 0) + int(round(s.probability * shots))
                    
                    total = sum(counts.values())
                    assert abs(total - shots) / total < 1e-12, "Sum of the counts is not equal to the number of shots!!"
                    dict_counts[f"RC_circuit_{ri}"] = counts

                self.measurement_counts[f"job_{ind_group}"] = dict_counts
                pp_results.append(Result(value=np.inf))

        if self.obs_list[0] is None: 
            self.save_data(exp_label=self.exp_name, save_results=True)

        return BatchResult(results=pp_results)

    
    def save_data(self, batch=None, exp_label=None, save_batch=False, save_twirls=True, save_results=False):
        """
        Save twirling indices AND (optionally) the batch of jobs..
        """
        import numpy as np
        from pathlib import Path

        # Base output directory
        out_dir = Path(self.folder_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Experiment-specific subdirectory
        exp_dir = out_dir / (
            f"{exp_label}_circuit_twirls_"
            f"ns{self.samples_for_naming}_"
            f"ntrotter{self.trotter_steps}_"
            f"seed{self.rng_seed}"
        )
        circ_dir = out_dir   # = get_circuit_folder(config, top_folder=False), passed in as folder_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        circ_dir.mkdir(parents=True, exist_ok=True)
        
        if not save_results: 
            # Save twirling indices
            if save_twirls:
                twirl_path = circ_dir / "twirl_indices_right.npy"
                np.save(twirl_path, self.twirl_indices_right)
            # Save the batch of jobs
            if save_batch:
                batch_filename = f"{circ_dir}/batch_jobs_Nt{self.trotter_steps}_{self.batch_label}"
                for ji, job in enumerate(batch.jobs):
                    Batch(jobs=[job]).dump(f"{batch_filename}_sample_{ji}")
            
        else:
            assert self.measurement_counts, "The measurement dictionary is empty!!"
            results_path = exp_dir / "results.json"
            serializable_results = {
                str(k): {
                    str(rk): {str(bk): int(bv) for bk, bv in rc_counts.items()}
                    for rk, rc_counts in rc_dict.items()
                }
                for k, rc_dict in self.measurement_counts.items()
            }
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=4)


class cRCCNOT_plugin(RCCNOT_plugin):
    '''This does full RC (cross-talk + local RC)'''
    def __init__(self, rng_seed=0, nsamples=4, ntrotter=None, circuit_og=None, coherent_noise=False,
                 coherent_noise_angles=None, turn_off_RC=False, same_samples_rc_nec=True, log_batch=False,
                 log_twirls=False, exp_name=None, exact_twirling=False, batch_label=None,
                 neighbor_noise=False, neighbor_noise_angles=None, turn_off_cRC=False,
                 turn_off_error_mitigation=False, folder_name="datafiles",
                 randomize_nec_input_state=True, run_nec_obs_job=False, LNN=False, verbose=False):

        # Call parent __init__
        super().__init__(rng_seed=rng_seed, nsamples=nsamples, ntrotter=ntrotter, circuit_og=circuit_og,
                         coherent_noise=coherent_noise, coherent_noise_angles=coherent_noise_angles,
                         turn_off_RC=turn_off_RC, same_samples_rc_nec=same_samples_rc_nec, log_batch=log_batch,
                         log_twirls=log_twirls, exp_name=exp_name, exact_twirling=exact_twirling,
                         batch_label=batch_label, folder_name=folder_name,
                         randomize_nec_input_state=randomize_nec_input_state,
                         run_nec_obs_job=run_nec_obs_job, LNN=LNN, verbose=verbose)
        
        self.neighbor_noise = neighbor_noise
        self.neighbor_noise_angles = neighbor_noise_angles
        self.n_neighbors = circuit_og.nbqbits - 2  
        self.twirl_neighbor_P = None
        self.twirl_neighbor_R = None
        self.neighbor_map = {
            (0, 1): 2,
            (1, 0): 2,
            (0, 2): 1,
            (2, 0): 1,
            (1, 2): 0,
            (2, 1): 0,
        }
        self.turn_off_cRC = turn_off_cRC
        self.turn_off_error_mitigation = turn_off_error_mitigation
        if self.turn_off_error_mitigation:
            self.turn_off_RC  = True
            self.turn_off_cRC = True
            self.Pauli2_keys  = ["00"]*16
            self.gid_label    = "nec"
            if verbose: print("Building jobs for NEC")
        elif (not turn_off_RC) and turn_off_cRC:
            if verbose: print("Building jobs for RC+NEC")
            self.gid_label = "rcc"
        else:
            if verbose: print("Building jobs for cRC+NEC")
            self.gid_label = "crcc"


    def compile(self, batch, specs=None):
        njobs = len(batch.jobs)
        rng   = np.random.default_rng(self.rng_seed)

        # ── Active-qubit twirl indices (same as parent) ──────────────────────
        if self.same_samples_rc_nec:
            base = rng.integers(0, 16, size=(self.nsamples, self.n_CNOTs))
            self.twirl_indices_right = np.repeat(base[None, :, :], njobs, axis=0)
        else:
            self.twirl_indices_right = rng.integers(0, 16, size=(njobs, self.nsamples, self.n_CNOTs))

        # ── Neighbor twirl indices (cRC addition) ────────────────────────────
        if self.same_samples_rc_nec:
            base_P = rng.integers(0, 4, size=(self.nsamples, self.n_CNOTs, self.n_neighbors))
            base_R = rng.integers(0, 3, size=(self.nsamples, self.n_CNOTs, self.n_neighbors))
            self.twirl_neighbor_P = np.repeat(base_P[None, :, :, :], njobs, axis=0)
            self.twirl_neighbor_R = np.repeat(base_R[None, :, :, :], njobs, axis=0)
        else:
            self.twirl_neighbor_P = rng.integers(0, 4, size=(njobs, self.nsamples, self.n_CNOTs, self.n_neighbors))
            self.twirl_neighbor_R = rng.integers(0, 3, size=(njobs, self.nsamples, self.n_CNOTs, self.n_neighbors))

        # ── Job expansion loop (same structure as parent) ────────────────────
        self.obs_list   = []
        self.shots_list = []
        new_cRC_jobs    = []

        for ji, job in enumerate(batch.jobs):
            if not self.turn_off_error_mitigation:
                self.obs_list.append(job.observable)
                self.shots_list.append(job.nbshots)

            assert job.meta_data is not None
            job_role = job.meta_data["role"]
            for si in tqdm(range(self.nsamples), desc=f"Nt={self.trotter_steps} {self.batch_label} {job_role} twirls",
                           leave=False, mininterval=0.5):
                if not self.cohr_noise:
                    new_circuit = self.twirl_CNOTs(job.circuit, index=(ji, si), role=job_role)
                else:
                    new_circuit = self.twirl_dCNOTs(job.circuit, index=(ji, si), role=job_role)

                if self.run_nec_obs_job and job_role.endswith("_nec") and self.nsamples > 1 and self.randomize_nec_input_state and job.observable is not None:
                    input_bitstring  = sample_index_to_input_bitstring(si, self.nsamples, self.nqubits)
                    target_bitstring = job.meta_data["classical_nec_map"][input_bitstring]
                    obs_for_job = build_projector_observable(target_bitstring, self.nqubits)
                else:
                    obs_for_job = job.observable

                new_job = new_circuit.to_job(observable=obs_for_job, nbshots=job.nbshots)
                new_job.meta_data = {}
                new_job.meta_data["role"] = job.meta_data["role"]
                new_job.meta_data["gid"]  = job.meta_data["gid"] + f"_{self.gid_label}{si}"
                new_cRC_jobs.append(new_job)

        batch_jobs = Batch(jobs=new_cRC_jobs)


        # -------- Correction of obs_list and shots_list -----------------
        if self.turn_off_error_mitigation:
            self.obs_list   = [batch.jobs[0].observable, batch.jobs[njobs//2].observable] 
            self.shots_list = [batch.jobs[0].nbshots,    batch.jobs[njobs//2].nbshots]
        if self.turn_off_error_mitigation and njobs==1:
            self.obs_list   = [batch.jobs[0].observable] 
            self.shots_list = [batch.jobs[0].nbshots]

        if self.log_data or self.log_twirls:
            self.save_data(batch_jobs, save_batch=self.log_data,
                           save_twirls=self.log_twirls, exp_label=self.exp_name)
        return batch_jobs

    def twirl_CNOTs(self, circuit, index, role=None):
        '''
        Replaces the CNOTS by custom crcCNOT gates in the circuit
        
        :param circuit: Circuit to replace CNOTS
        :param index: a tuple containing indices of job type, and samples

        Returns a circuit object
        '''
        prog = Program()
        q = prog.qalloc(circuit.nbqbits)

        assert role is not None, "role cannot be None!!"
        if role.endswith("_nec") and self.nsamples > 1 and self.randomize_nec_input_state:
            input_bitstring = sample_index_to_input_bitstring(index[1], self.nsamples, self.nqubits)
            prep_computational_basis_input(prog, q, input_bitstring)

        assert index[1]<self.nsamples, "The sample_index is invalid!!"

        list_indices   = self.twirl_indices_right[index[0], index[1], :]
        list_indices_P = self.twirl_neighbor_P[index[0], index[1], :]
        list_indices_R = self.twirl_neighbor_R[index[0], index[1], :]

        assert len(list_indices)   == self.n_CNOTs, "The length of twirling indices should be the same as the number of CNOTs!"
        assert len(list_indices_P) == self.n_CNOTs, "The length of twirling indices should be the same as the number of CNOTs!"
        assert len(list_indices_R) == self.n_CNOTs, "The length of twirling indices should be the same as the number of CNOTs!"
        
        count_CNOTs = 0
        count_2q_out = 0
        for item in circuit.iterate_simple():
            gate, qbits, params = unpack(item)
            name = getattr(gate, "name", None)
            gname = name if name is not None else gate

            if gname == "CNOT":
                assert len(qbits)==2, "CNOT must act on 2 qubits"
                key      = self.Pauli2_keys[list_indices[count_CNOTs]]
                index_P  = list_indices_P[count_CNOTs][0]
                index_R  = list_indices_R[count_CNOTs][0]

                neighbor = self.neighbor_map[(qbits[0], qbits[1])]
                prog.apply(crcCNOT(key, index_P, index_R, coherent_noise=False, angles=None, neighbor_noise=False, neighbor_noise_angles=None, turn_off_cRC=self.turn_off_cRC), q[qbits[0]], q[qbits[1]], q[neighbor])
                count_CNOTs += 1
                count_2q_out+= 1
                continue
            
            # if gate already object other than CNOT, apply
            elif hasattr(gate, "arity"):
                prog.apply(gate, *[q[i] for i in qbits])
                if not isinstance(gate, str) and hasattr(gate, "arity") and gate.arity == 2:
                    count_2q_out += 1
                continue
            
            # gate is a string name: build the gate object
            elif not isinstance(gate, str):
                raise TypeError(f"Unexpected gate type: {type(gate)}")

            if gate in PARAM_1Q:
                theta = params[0] if isinstance(params, (list, tuple)) else params
                prog.apply(PARAM_1Q[gate](theta), q[qbits[0]])
            elif gate in NAMED_1Q:
                prog.apply(NAMED_1Q[gate], q[qbits[0]])
            else:
                raise RuntimeError(f"Unsupported string gate {gate!r} (need a resolver for it).")
        
        assert count_CNOTs==self.n_CNOTs, "No. of CNOTS is not consistent!!"
        assert count_2q_out==self.n_CNOTs, "No. of CNOTS is not consistent!!"
        twirled_circuit = prog.to_circ()
        return twirled_circuit


    def twirl_dCNOTs(self, circuit, index, role=None):
        '''
        Replaces the CNOTS or noisy_CNOTs by custom crcCNOT gates in the circuit
        
        :param circuit: Circuit to replace CNOTS
        :param index: a tuple containing indices of job type, and samples

        Returns a circuit object
        '''
        prog = Program()
        q = prog.qalloc(circuit.nbqbits)

        assert role is not None, "role cannot be None!!"
        if role.endswith("_nec") and self.nsamples > 1 and self.randomize_nec_input_state:
            input_bitstring = sample_index_to_input_bitstring(index[1], self.nsamples, self.nqubits)
            prep_computational_basis_input(prog, q, input_bitstring)

        assert index[1]<self.nsamples, "The sample_index is invalid!!"

        list_indices   = self.twirl_indices_right[index[0], index[1], :] 
        list_indices_P = self.twirl_neighbor_P[index[0], index[1], :]
        list_indices_R = self.twirl_neighbor_R[index[0], index[1], :]

        assert len(list_indices)   == self.n_CNOTs, "The length of twirling indices should be the same as the number of CNOTs!"
        assert len(list_indices_P) == self.n_CNOTs, "The length of twirling indices should be the same as the number of CNOTs!"
        assert len(list_indices_R) == self.n_CNOTs, "The length of twirling indices should be the same as the number of CNOTs!"

        count_CNOTs = 0
        count_2q_out = 0
        for op in circuit.ops:
            gdef = circuit.gateDic.get(op.gate) if isinstance(op.gate, str) else None
            if gdef:
                gname = gdef.syntax.name
                syntax_params = gdef.syntax.parameters
                arity = gdef.arity
            qbits = op.qbits

            param_list = []
            for p in syntax_params or []:
                if getattr(p, "double_p", None) is not None:
                    param_list.append(float(p.double_p))
                elif getattr(p, "int_p", None) is not None:
                    param_list.append(float(p.int_p))
                else:
                    param_list.append(None)

            if gname in ["dCNOT", "CNOT"]:
                assert arity==2, "CNOT must act on 2 qubits"
                key = self.Pauli2_keys[list_indices[count_CNOTs]]
                if gname=="dCNOT":
                    assert np.linalg.norm(np.array(self.cohr_noise_angles) - np.array(param_list)) < 1e-16, "Angles of dCNOT are not consistent!!"
                index_P = list_indices_P[count_CNOTs][0]
                index_R = list_indices_R[count_CNOTs][0]   

                neighbor = self.neighbor_map[(qbits[0], qbits[1])]
                prog.apply(crcCNOT(key, index_P, index_R, coherent_noise=self.cohr_noise, angles=self.cohr_noise_angles, neighbor_noise=self.neighbor_noise, neighbor_noise_angles=self.neighbor_noise_angles, turn_off_cRC=self.turn_off_cRC), q[qbits[0]], q[qbits[1]], q[neighbor])
                count_CNOTs += 1
                count_2q_out+=1
                continue
            elif arity==2:
                raise RuntimeError(f"Don't know how to apply gate {gname!r} (id={op.gate!r})")
            
            if gname in PARAM_1Q:
                theta = param_list[0] if isinstance(param_list, (list, tuple)) else param_list
                prog.apply(PARAM_1Q[gname](theta), q[qbits[0]])
            elif gname in NAMED_1Q: 
                prog.apply(NAMED_1Q[gname], q[qbits[0]])
            else:
                raise RuntimeError(f"Unsupported string gate {gname!r} (need a resolver for it).")
        
        assert count_CNOTs==self.n_CNOTs, "No. of CNOTS is not consistent!!"
        assert count_2q_out==self.n_CNOTs, "No. of CNOTS is not consistent!!"
        twirled_circuit = prog.to_circ()
        return twirled_circuit

    # post_process  ← NOT defined here, falls through to RCCNOT_plugin.post_process
    # save_data     ← NOT defined here, falls through to RCCNOT_plugin.save_data


class NEC_plugin(AbstractPlugin):
    def __init__(self, circuit_og=None, rng_seed=0, log_batch=False,
                 exp_name=None, artificial_noise=None,
                 folder_name="datafiles",
                 split_jobs=False, nsamples=None,
                 measurement_counts=False, depth_label=None,
                 compute_classical_nec_map=False, verbose=False):
        super().__init__()

        assert circuit_og is not None

        self.nqubits = circuit_og.nbqbits
        self.rng_seed = rng_seed
        self.log_data = log_batch
        self.verbose = verbose
        self.exp_name = exp_name
        self.coherent_noise_angles = artificial_noise
        self.nec_fid_list = []
        self.folder_name = folder_name
        self.split_jobs = split_jobs
        self.nsamples   = nsamples if nsamples is not None else 1
        if self.split_jobs:
            assert self.nsamples is not None,"nsamples cannot be None!!"
        self.measurement_counts = measurement_counts
        self.job_meta_data_roles = []
        self.classical_nec_map = {}
        self.compute_classical_nec_map = compute_classical_nec_map
        self.depth_label = depth_label
        if self.measurement_counts:
            assert depth_label is not None, "depth_label cannot be None!!"

    def build_nec(self, circuit):
        '''
        Builds a CNOT skeleton of a given circuit
        
        :param circuit: Input circuit

        Returns a circuit object
        '''
        prog = Program()
        q = prog.qalloc(circuit.nbqbits)

        count_CNOTs = 0
        for item in circuit.iterate_simple():
            gate, qbits, params = unpack(item)
            name = getattr(gate, "name", None)
            gname = name if name is not None else gate
            
            if gname in ["CNOT", "dCNOT"]:
                assert len(qbits)==2, "CNOT must act on 2 qubits"

                if self.coherent_noise_angles is None:
                    prog.apply(CNOT, q[qbits[0]], q[qbits[1]])
                else:
                    prog.apply(noisy_cnot(*self.coherent_noise_angles), q[qbits[0]], q[qbits[1]])
                count_CNOTs += 1
                continue
                    
        assert count_CNOTs%2==0, "The CNOT skeleton is not identity!!"

        CNOT_skeleton_circuit = prog.to_circ()
        return CNOT_skeleton_circuit


    def compile(self, batch, specs=None):
        '''
        Produces a new batch of compiled circuits from a given batch of circuits.
        
        :param batch: A batch of circuits - A batch of original_circuits.

        Returns a new batch of compiled circuits - [Original_circuits, NEC_circuits]
        '''
        njobs = len(batch.jobs)
        uids = [f"job_{nj}" for nj in np.arange(njobs)]
        self.job_meta_data_roles = [None]*njobs

        full_job_list = []
        for index, job in enumerate(batch.jobs):
            if self.verbose: print(f"Sending Job #{index}")

            gid = uids[index]
            org_job = copy.deepcopy(job) 
            org_job.meta_data = {}
            org_job.meta_data["gid"] = gid
            org_job.meta_data["role"] = job.meta_data["role"]

            v00 = np.zeros(2**self.nqubits, dtype=complex)  # |00...0>
            v00[0] = 1
            M = np.outer(v00, v00.conj())                  # |0...0><0...0|  (2^nx2^n)
            fid_obs = Observable(self.nqubits, matrix=M)
            nec_circuit = self.build_nec(job.circuit)
            classical_nec_map_for_this_job = classically_simulate_skeleton_qlm(nec_circuit, self.nqubits)

            if self.measurement_counts and self.compute_classical_nec_map:
                self.classical_nec_map[index] = classical_nec_map_for_this_job

            nec_obs = None if self.measurement_counts else fid_obs
            nec_job = nec_circuit.to_job(observable=nec_obs, nbshots=job.nbshots)
            nec_job.meta_data = {}
            nec_job.meta_data["gid"] = gid
            nec_job.meta_data["role"] = job.meta_data["role"]+"_nec"
            nec_job.meta_data["classical_nec_map"] = classical_nec_map_for_this_job
            if not self.split_jobs:
                full_job_list += [org_job, nec_job]
            else:
                full_job_list += split_job_by_shots(org_job, n_samples=self.nsamples) + split_job_by_shots(nec_job, n_samples=self.nsamples)
            
            self.job_meta_data_roles[index] = [org_job.meta_data["role"]]*self.nsamples + [nec_job.meta_data["role"]]*self.nsamples

        # Return full compiled batch
        batch_jobs = Batch(jobs=full_job_list)
        if self.compute_classical_nec_map: self.save_data(save_classical_map=True, exp_label=self.exp_name)
        return batch_jobs

    def post_process(self, batch_result):
        '''
        Gives error corrected values of observables.

        :param batch_result: BatchResult object. BatchResult.results is the list of all (job+nec)-results (2*njobs)

        Returns a new BatchResult object, with len(BatchResult.results) = njobs 
        ''' 
        assert self.nsamples == 1, "This only works when there is 1 copy of the NEC circuit!!"
        grouped_job_nec_results = [
            batch_result.results[i:i+2*self.nsamples]
            for i in range(0, len(batch_result.results), 2*self.nsamples)
        ]

        final_results = []
        assert len(grouped_job_nec_results) == len(self.job_meta_data_roles), "Consistency check regarding the number of original jobs!!"
        for ind_group, results in enumerate(grouped_job_nec_results):

            assert len(results) == 2, "Shape of grouped_results is incorrect!!"

            nec_fid = results[1].value if not self.measurement_counts else np.nan# The fidelity is computed irrespective of whether the original job is a sampling job!!
            self.nec_fid_list.append(nec_fid)

            if self.verbose: print("  [Build] fidelity of noisy (or RC) CNOT skeleton: ", nec_fid)

            if not self.measurement_counts:
                obs_job = results[0].value
                if self.verbose: print("  [Build] Obs value with raw/noisy (or RC) circuit: ", obs_job)
                error_corrected_obs = obs_job/nec_fid
            else:
                error_corrected_obs = np.inf

            final_results.append(Result(value=error_corrected_obs))
        return BatchResult(results=final_results)

    def save_data(self, batch=None, exp_label=None, save_batch=False, nec_results=False, save_classical_map=False):
        """
        Write only the classical NEC map (nec_classical_map_{depth}.json), when save_classical_map is set;
        save_batch and nec_results are ignored.
        """
        from pathlib import Path

        exp_dir = Path(self.folder_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        if save_classical_map and self.classical_nec_map:
            serializable_map = {
                str(job_idx): input_output_map
                for job_idx, input_output_map in self.classical_nec_map.items()
            }
            with open(exp_dir / f"nec_classical_map_{self.depth_label}.json", "w", encoding="utf-8") as f:
                json.dump(serializable_map, f, indent=4)            


class RC_NEC_post_compile(AbstractPlugin):
    def __init__(self, obs_list, shots_list, exp_name, nqubits, samples_per_mitigation_strategy, rng_seed=0, job_tags=None, folder_name="datafiles", chunk_samples_per_mitigation_strategy=None, chunk_index=0, ndepths=1, nshots=0, ems_list=None, infidelity_from_counts=None):
        assert ems_list, "ems_list cannot be empty!!"
        assert infidelity_from_counts is not None, "It has to be a boolean variable!!"

        self.nqubits            = nqubits
        self.obs_list           = obs_list
        self.shots_list         = shots_list
        self.nsamples_per_mitigation_strategy = samples_per_mitigation_strategy 
        self.exp_name           = exp_name
        self.rng_seed           = rng_seed
        self.measurement_counts = {}
        self.fidelity_nec_dict  = {}
        self.measure_infidelity_from_counts = infidelity_from_counts
        self.job_tags           = job_tags
        self.folder_name        = folder_name
        self.ndepths            = ndepths
        self.nshots             = nshots
        self.num_EMS            = len(ems_list)

        if chunk_samples_per_mitigation_strategy is None:
            self.chunk_samples_per_mitigation_strategy = samples_per_mitigation_strategy
            self.chunk_index = 0
        else:
            self.chunk_samples_per_mitigation_strategy = chunk_samples_per_mitigation_strategy 
            self.chunk_index = chunk_index
            assert self.chunk_index is not None

    def compile(self, batch, specs=None):
        return batch

    def post_process(self, batch_result):
        '''
        Post-process the results of each job type (rc, rc+nec, crc, crc+nec ...). Each group's raw bitstring counts (one dict per randomized compiled circuit sample) are stored, not averaged: NEC groups in self.fidelity_nec_dict, the other groups in self.measurement_counts. Every group returns np.inf as its value.

        :param batch_result: BatchResult object. BatchResult.results is the list of all results (nsamples*njobs)

        Returns a new BatchResult object, with len(BatchResult.results) = njobs 
        '''        
        grouped_results = [
            batch_result.results[i:i+self.chunk_samples_per_mitigation_strategy]
            for i in range(0, len(batch_result.results), self.chunk_samples_per_mitigation_strategy)# chunk_samples_per_mitigation_strategy = total samples per EMS/n_chunks (= 1)
        ]    # A List of lists


        n_EMS          = self.num_EMS #int(round(len(self.obs_list)/self.ndepths))        
        n_groups       = len(grouped_results)  # = 6 (one per EMS role)
        
        if not self.obs_list:
            if self.measure_infidelity_from_counts:
                self.obs_list = [None]*n_EMS*self.ndepths
            else:
                assert self.obs_list, "Empty list"
        assert n_groups % n_EMS == 0, f"n_groups ({n_groups}) not divisible by n_EMS ({n_EMS})"
        obs_chunk      = list(self.obs_list  [0 : n_EMS]) * (n_groups // n_EMS)
        shots_chunk    = list(self.shots_list[0 : n_EMS]) * (n_groups // n_EMS)
        assert (len(obs_chunk) == n_groups) and (len(shots_chunk) == n_groups), \
            f"obs_chunk ({len(obs_chunk)}) != grouped_results ({n_groups})"
        if self.job_tags is not None:
            assert len(self.job_tags) == n_groups, \
                f"length of job_tags ({len(self.job_tags)}) != grouped_results ({n_groups})"

        pp_results = []
        for ind_group,results in enumerate(grouped_results): # ind_group = 0 to chunk_size-1
            avg_value = 0
            job_tag = self.job_tags[ind_group] if self.job_tags is not None else None
            is_nec_group = job_tag is not None and job_tag[-3:] == "nec"
            key_for_job = f"job_{job_tag}_ems{ind_group}" if job_tag is not None else f"job_{ind_group}" # ind_group goes from 0 to chunk_size-1

            if is_nec_group: 
                # ---- counts-derived fidelity: store raw bitstring counts ----
                self.measure_infidelity_from_counts = True
                shots = self.nshots
                if shots == 0: shots = 1e14
                if getattr(self, "verbose", False): print(f"  [Build] group {ind_group}, job_tag={job_tag}, shots={shots}, len(results)={len(results)}")
                dict_counts = {}

                for ri, result in enumerate(results): # ri -> sample index in each chunk = 0 - does not go further as we have 1 sample per chunk
                    counts = {}
                    for s in result:
                        state = int.from_bytes(s._state, "big")
                        bitstring = format(state, f"0{self.nqubits}b")
                        counts[bitstring] = counts.get(bitstring, 0) + int(round(s.probability * shots))

                    total = sum(counts.values())
                    assert abs(total - shots) / total < 1e-12, "Sum of the counts is not equal to the number of shots!!"

                    key_for_circuits = f"{job_tag}_circuit_{ri}" if job_tag is not None else f"NEC_circuit_{ri}" # ri -> sample index in each chunk = 0
                    dict_counts[key_for_circuits] = counts

                self.fidelity_nec_dict[key_for_job] = dict_counts
                pp_results.append(Result(value=np.inf))

            else:
                shots = self.nshots
                if shots == 0: shots = 1e14
                if getattr(self, "verbose", False): print(f"  [Build] group {ind_group}, job_tag={job_tag}, shots={shots}, len(results)={len(results)}")
                dict_counts = {} # Each group of RC jobs (with a given job id) - for a given circuit, has the same dictionary
                
                for ri,result in enumerate(results): # iterate over RC sample circuit results
                    counts = {}
                    for s in result:
                        state = int.from_bytes(s._state, "big")
                        bitstring = format(state, f"0{self.nqubits}b")
                        counts[bitstring] = counts.get(bitstring, 0) + int(round(s.probability * shots))
                    
                    total = sum(counts.values())
                    assert abs(total - shots) / total < 1e-12, "Sum of the counts is not equal to the number of shots!!"

                    key_for_circuits = f"{job_tag}_circuit_{ri}" if job_tag is not None else f"RC_circuit_{ri}"
                    dict_counts[key_for_circuits] = counts

                self.measurement_counts[key_for_job] = dict_counts
                pp_results.append(Result(value=np.inf))

        if self.fidelity_nec_dict:
            self.save_data(exp_label=self.exp_name, save_fidelity_nec=True)
        if self.measurement_counts:
            self.save_data(exp_label=self.exp_name, save_results=True)

        return BatchResult(results=pp_results)

    
    def save_data(self, batch=None, exp_label=None, save_results=False, save_fidelity_nec=False, job_tag=None):
        """
        Results for respective jobs are saved one by one in different dictionaries
        """
        from pathlib import Path
        if getattr(self, "verbose", False): print(f"  [Build] Chunk index: {self.chunk_index}")
        # Base output directory
        out_dir = Path(self.folder_name) / "raw_results"
        out_dir.mkdir(parents=True, exist_ok=True)


        if save_results:
            results_path = out_dir / "results_counts.json"
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = {}
                
            serializable_results = {
                str(k): {
                    str(rk): {str(bk): int(bv) for bk, bv in rc_counts.items()}
                    for rk, rc_counts in rc_dict.items()
                }
                for k, rc_dict in self.measurement_counts.items()
            }
            chunk_key = f"chunk_{self.chunk_index}"  # pass chunk_index into post-plugin
            existing[chunk_key] = serializable_results  # nest under chunk key
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4)
        
        if save_fidelity_nec:
            if not self.measure_infidelity_from_counts:
                fidelity_path = out_dir / "results_fidelity.json"
                if fidelity_path.exists():
                    with open(fidelity_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                else:
                    existing = {}

                chunk_key = f"chunk_{self.chunk_index}"  # pass chunk_index into post-plugin
                existing[chunk_key] = self.fidelity_nec_dict  # nest under chunk key

                with open(fidelity_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=4)

            else:
                nec_results_path = out_dir / "results_fidelity.json"
                if nec_results_path.exists():
                    with open(nec_results_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                else:
                    existing = {}

                serializable_nec_results = {
                    str(k): {
                        str(rk): {str(bk): int(bv) for bk, bv in rc_counts.items()}
                        for rk, rc_counts in rc_dict.items()
                    }
                    for k, rc_dict in self.fidelity_nec_dict.items()
                }
                chunk_key = f"chunk_{self.chunk_index}"
                existing[chunk_key] = serializable_nec_results
                with open(nec_results_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=4)


def interleave_the_batch_multidepth_from_disk(config,
                                              t_list,
                                              n_rc_samples,
                                              batch_names=None,
                                              verbose=False):
    '''
    Loads one depth at a time from disk and appends that depth's jobs for each chunk to the
    chunk_jobs list in interleaved order; frees only the batch file handles, not the jobs. Never holds all depths in RAM.
    '''
    if batch_names is None:
        batch_names = ["NEC", "RC+NEC", "cRC+NEC"]

    BATCH_NAMES = batch_names


    n_depths = len(t_list)
    n_roles  = 2*len(BATCH_NAMES)
    n_samples_per_role = config["samples"]
    assert n_samples_per_role==n_rc_samples

    n_total  = n_depths * n_roles * n_samples_per_role
    

    chunk_size = config["chunk_size"]
    assert chunk_size == n_depths * n_roles, \
    f"chunk_size ({chunk_size}) must equal n_depths*n_roles ({n_depths*n_roles}={n_depths}*{n_roles})"
    n_chunks = int(round(n_total/chunk_size))

    print(f"  [Build] {n_depths} depths, {n_roles} roles, {n_rc_samples} RC samples: "
          f"{n_total} jobs in {n_chunks} chunks of {chunk_size}")

    def create_an_interleaved_chunk(nc):
        chunk_jobs = []
        
        for depth_idx, t_circuit in enumerate(t_list):
            nt = int(round(t_circuit / config["dt"]))
            config["Nt"] = nt
            config["T"]  = t_circuit

            for batch_name in BATCH_NAMES:
                batch_filename = os.path.join(
                    get_circuit_folder(config, top_folder=False),
                    f"batch_jobs_Nt{nt}_{batch_name}"
                )
                # Load individual sample files instead of full batch
                b_rc    = Batch.load(f"{batch_filename}_sample_{nc}.batch")
                b_rcnec = Batch.load(f"{batch_filename}_sample_{nc+n_samples_per_role}.batch")
                collect_jobs = [b_rc.jobs[0], b_rcnec.jobs[0]]
                chunk_jobs.extend(collect_jobs)
                del b_rc, b_rcnec
                gc.collect()
        assert len(chunk_jobs)==chunk_size,"Wrong size of chunk_jobs!!"

        return chunk_jobs
    # Save chunks to disk one at a time

    chunk_paths = []
    myqlm_chunk_folder = get_chunk_folder(get_circuit_folder(config, top_folder=True), "myqlm")
    for nc in tqdm(range(n_chunks), desc="Interleaving chunks", mininterval=0.5):
        # ======================== Create the chunks directly =======================
        chunk_path = os.path.join(
            myqlm_chunk_folder,
            f"batch_jobs_interleaved_chunk{nc}"
        )
        chunk_paths.append(chunk_path)

        if os.path.exists(f"{chunk_path}.batch"):
            continue

        chunk_jobs = create_an_interleaved_chunk(nc)
        bc = Batch(jobs=chunk_jobs)
        bc.dump(chunk_path)
        del bc
        gc.collect()

        if verbose: print(f"  [Chunk] Saved chunk {nc+1}/{n_chunks} to disk.")

    return chunk_paths

class HardwareSubmission_plugin(AbstractPlugin):
    """
    Interleaves raw, raw_nec, rc, rc_nec, crc, crc_nec jobs for hardware submission.
    
    Interleaving pattern (per rc_sample, per depth): one job of each role, in the order of self.roles
    (raw, raw_nec, rc, rc_nec, crc, crc_nec).
    
    On the way back, reshuffles results into original order
    before passing to RC_NEC_post_compile.
    """

    def __init__(self, n_rc_samples, n_shots_per_sample, 
                 is_the_batch_interleaved=False, folder_name=None, 
                 results_folder="datafiles", 
                 chunk_samples_rc=None, chunk_id=None, 
                 chunk_size=None, ndepths=1, roles=None):
        self.n_rc_samples             = int(n_rc_samples)
        self.n_shots_per_instance     = int(n_shots_per_sample)
        self.job_map                  = None   # stores (original_index, job_type) for reshuffling
        self.is_the_batch_interleaved = is_the_batch_interleaved
        self.folder_name              = folder_name 
        self.results_folder           = results_folder
        self.ndepths                  = ndepths
        self.roles = roles if roles is not None else ["raw", "raw_nec", "rc", "rc_nec", "crc", "crc_nec"]

        if chunk_samples_rc is None: # No chunking
            self.chunk_samples_rc = int(n_rc_samples)
            self.chunk_id         = 0
        else:
            self.chunk_samples_rc = chunk_samples_rc
            assert chunk_id      != None
            self.chunk_id         = chunk_id
            self.chunk_size       = chunk_size

    def compile(self, batch, specs):
        if self.is_the_batch_interleaved:
            if len(batch.jobs) % self.chunk_samples_rc != 0:
                print("The batch is not properly interleaved or chunked!! Please turn off this option, or redefine the chunk size.")
    
            roles = self.roles 
            self.job_map = []
            for depth_idx in range(self.ndepths):
                for role_idx, role in enumerate(roles):
                    local_index  = depth_idx * len(roles) + role_idx
                    global_index = self.chunk_id * self.chunk_size + local_index
                    self.job_map.append((global_index, role, depth_idx, self.chunk_id))
            return batch
        else:
            assert self.chunk_samples_rc == self.n_rc_samples, "Only works without chunking!!"

    def post_process(self, batch_result):
        """Reshuffle results back into original order, 
        grouped by depth: RAW, RAW_NEC, RC, RC_NEC, CRC, CRC_NEC."""
        
        n_depths = self.ndepths #max(depth_idx for _, _, depth_idx, _ in self.job_map) + 1
        roles    = self.roles #["raw", "raw_nec", "rc", "rc_nec", "crc", "crc_nec"]
    
        # Initialize per-depth result lists
        depth_results = {
            d: {role: [] for role in roles}
            for d in range(n_depths)
        }
    
        chunk_start = self.chunk_id * self.chunk_size
        for index, role, depth_idx, sample_idx in self.job_map:
            result = batch_result.results[index - chunk_start]
            depth_results[depth_idx][role].append(result)

    
        # Concatenate results depth by depth, role by role per chunk
        all_results = []
        for d in range(n_depths):
            for role in roles:
                all_results += depth_results[d][role]
    
        return BatchResult(results=all_results)

    
    def save_batch(self, batch):

        # Base output directory
        out_dir = Path(self.results_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Experiment-specific subdirectory
        exp_dir = out_dir / (
            f"{self.folder_name}"
        )

        exp_dir.mkdir(parents=True, exist_ok=True) 

        batch_filename = f"{exp_dir}/batch_jobs_interleaved"
        batch.dump(batch_filename)

        job_map_dict = {}
        job_map_dict["job_map"] = self.job_map
        job_map_path = f"{exp_dir}/job_map.json"
        with open(job_map_path, "w", encoding="utf-8") as f:
            json.dump(job_map_dict, f, indent=4)
       

   
def build_trotter_step_paper_IBM():
    nsites = 3
    q_routine = QRoutine()
    nqubits = nsites
    q = q_routine.new_wires(nqubits)
    assert nqubits==3, "The paper had 3 qubits!"
    print("  [Build] Using the circuit from the paper.")
    q_routine.apply(RZ(0.689),  q[0])
    q_routine.apply(RZ(-2.85),  q[1])
    q_routine.apply(RZ(-0.3),   q[2])

    # --- Two-qubit block between q0 (control) and q1 (target)  [big red box]
    # q0 line: √X, Rz(-pi/2), CNOT, √X, Rz(-0.1), √X, CNOT, Rz(-pi/2), √X, Rz(-0.189)
    # q1 line: √X,          (target),     Rz(0.1),     (target), √X, Rz(2.95)
    q_routine.apply(sqrt_X, q[0])
    q_routine.apply(RZ(-np.pi/2), q[0])

    q_routine.apply(sqrt_X, q[1])
    q_routine.apply(CNOT, q[0], q[1])          # control q0 -> target q1

    q_routine.apply(RZ(0.1), q[1])

    q_routine.apply(sqrt_X, q[0])
    q_routine.apply(RZ(-0.1), q[0])
    q_routine.apply(sqrt_X, q[0])

    q_routine.apply(CNOT, q[0], q[1])          # control q0 -> target q1

    q_routine.apply(RZ(-np.pi/2), q[0])
    q_routine.apply(sqrt_X, q[0])
    q_routine.apply(RZ(-0.189), q[0])

    q_routine.apply(sqrt_X, q[1])
    q_routine.apply(RZ(2.95), q[1])

    # --- Right-side two-qubit block between q2 (control) and q1 (target)
    # (This is the big black rectangle on the right.)
    #
    # q1 line: Rz(-1.16), √X, Rz(-pi),  (target), Rz(0.1), (target)
    # q2 line: Rz(1.98),  √X, Rz(pi/2), (control), √X, Rz(-0.1), √X, (control)
    q_routine.apply(RZ(-1.16), q[1])
    q_routine.apply(sqrt_X, q[1])
    q_routine.apply(RZ(-np.pi), q[1])

    q_routine.apply(RZ(1.98), q[2])
    q_routine.apply(sqrt_X, q[2])
    q_routine.apply(RZ(np.pi/2), q[2])

    q_routine.apply(CNOT, q[2], q[1])          # control q2 -> target q1

    q_routine.apply(RZ(0.1), q[1])

    q_routine.apply(sqrt_X, q[2])
    q_routine.apply(RZ(-0.1), q[2])
    q_routine.apply(sqrt_X, q[2])

    q_routine.apply(CNOT, q[2], q[1])          # control q2 -> target q1

    q_routine.apply(RZ(-np.pi), q[1])
    q_routine.apply(sqrt_X, q[1])
    q_routine.apply(RZ(1.16), q[1])

    q_routine.apply(RZ(np.pi/2), q[2])
    q_routine.apply(sqrt_X, q[2])
    q_routine.apply(RZ(-1.98), q[2])

    q_routine.apply(RZ(-np.pi/2), q[0])
    q_routine.apply(sqrt_X, q[0])
    q_routine.apply(RZ(np.pi/2), q[0])

    q_routine.apply(RZ(np.pi/2), q[2])
    q_routine.apply(sqrt_X, q[2])
    q_routine.apply(RZ(-np.pi), q[2])

    q_routine.apply(CNOT, q[0], q[2])

    q_routine.apply(sqrt_X, q[0])
    q_routine.apply(RZ(-0.1), q[0])
    q_routine.apply(sqrt_X, q[0])

    q_routine.apply(RZ(0.1), q[2])

    q_routine.apply(CNOT, q[0], q[2])

    q_routine.apply(RZ(np.pi/2), q[0])
    q_routine.apply(sqrt_X, q[0])
    q_routine.apply(RZ(np.pi/2), q[0])

    q_routine.apply(RZ(-np.pi), q[2])
    q_routine.apply(sqrt_X, q[2])
    q_routine.apply(RZ(-np.pi/2), q[2])

    return q_routine        
    


def split_job_by_shots(job: Job, n_samples: int) -> list[Job]:
    """Split a job into n_samples copies that share its shots equally (nbshots == 0: n_samples plain copies)."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    if job.nbshots == 0:
        # nbshots==0 means "exact/unlimited": there is no finite shot budget.
        # This is only meaningful for testing against exact/analytic references;
        return [deepcopy(job) for _ in range(n_samples)]

    if job.nbshots < n_samples:
        raise ValueError(
            f"job.nbshots ({job.nbshots}) is smaller than n_samples "
            f"({n_samples}); cannot give every sub-job at least one shot."
        )

    base, remainder = divmod(job.nbshots, n_samples)
    assert remainder==0,"number of shots is not divisible by n_samples!"

    sub_jobs = []
    for _ in range(n_samples):
        shots_for_this_job = base
        sub_job = deepcopy(job)
        sub_job.nbshots = shots_for_this_job
        sub_jobs.append(sub_job)

    return sub_jobs
