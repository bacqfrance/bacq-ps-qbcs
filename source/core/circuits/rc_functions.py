import numpy as np
from qat.lang.AQASM import QRoutine, I, X, Y, Z, CNOT, RX, RY, RZ, AbstractGate
from qat.lang import build_gate

@build_gate("dCNOT", [float, float, float, float, float, float], arity=2)
def noisy_cnot(ax_c: float, ay_c: float, az_c: float,
               ax_t: float, ay_t: float, az_t: float):
    """
    2-qubit noisy CNOT with coherent single-qubit rotations.

    Parameters:
      (ax_c, ay_c, az_c): rotations on control qubit (q0)
      (ax_t, ay_t, az_t): rotations on target qubit (q1)
    """
    rout = QRoutine()
    q = rout.new_wires(2)

    # Ideal entangling gate
    rout.apply(CNOT, q[0], q[1])

    # Coherent noise on control (q0)
    rout.apply(RX(ax_c), q[0])
    rout.apply(RY(ay_c), q[0])
    rout.apply(RZ(az_c), q[0])

    # Coherent noise on target (q1)
    rout.apply(RX(ax_t), q[1])
    rout.apply(RY(ay_t), q[1])
    rout.apply(RZ(az_t), q[1])

    return rout



cnot_pauli_conj_table = {'00': '00',
 '01': '01',
 '02': '32',
 '03': '33',
 '10': '11',
 '11': '10',
 '12': '23',
 '13': '22',
 '20': '21',
 '21': '20',
 '22': '13',
 '23': '12',
 '30': '30',
 '31': '31',
 '32': '02',
 '33': '03'}

def rcCNOT(key, coherent_noise=False, angles=None):
    '''
    Pauli-twirled CNOT routine: the conjugated Paulis (from `cnot_pauli_conj_table[key]`) before the CNOT, the Paulis of `key` after it.

    :param key: str - "ij" 0 <= i,j <= 3
    '''
    ckey = cnot_pauli_conj_table[key]
    
    key = (int(key[0]), int(key[1]))
    ckey = (int(ckey[0]), int(ckey[1]))

    Paulis = [I, X, Y, Z]
    P_pre, Q_pre = Paulis[key[0]], Paulis[key[1]]
    P_pst, Q_pst = Paulis[ckey[0]], Paulis[ckey[1]]

    r = QRoutine()
    qc = r.new_wires(1)
    qt = r.new_wires(1)

    if P_pst is not I:
        r.apply(P_pst, qc)
    if Q_pst is not I:
        r.apply(Q_pst, qt)

    if coherent_noise:
        assert angles is not None, "angles cannot be NONE for a noisy gate!!"
    if coherent_noise and np.linalg.norm(angles) >= 1e-8:
        r.apply(noisy_cnot(*angles), qc, qt)
    else:
        r.apply(CNOT, qc, qt)

    if P_pre is not I:
        r.apply(P_pre, qc)
    if Q_pre is not I:
        r.apply(Q_pre, qt)

    return r

_PI2 = np.pi / 2
# Twirling gates for the neighboring qubits
T_P_GATES = [None, X, Y, Z]
T_R_GATES  = [RX(_PI2 ), RY(_PI2 ), RZ(_PI2 )]    # Rx, Ry, Rz
T_Rd_GATES = [RX(-_PI2), RY(-_PI2), RZ(-_PI2)]    # Rx, Ry, Rz
barrier = AbstractGate("BARRIER", [], arity=3,
                         matrix_generator=lambda: np.eye(8, dtype=complex))
def crcCNOT(active_key                   , 
            neighbor_P                   , 
            neighbor_R                   ,
            coherent_noise        = False, 
            angles                = None , 
            neighbor_noise        = False, 
            neighbor_noise_angles = None ,
            turn_off_cRC          = False
            ):
    """
    3-qubit cRC-CNOT      : wires are (control, target, neighbor).
 
    Active qubits         : standard Pauli twirl (the rcCNOT(active_key) gates, inlined below).
    Neighbor qubit        : T · Ri(π/2) |barrier| CNOT + noise |barrier| Ri(-π/2) · T
                            where T ∈ T_p = {I,X,Y,Z} (neighbor_P) and Ri ∈ T_R = {Rx,Ry,Rz} (neighbor_R).
 
    :param active_key     : str "ij", 0<=i,j<=3 — Pauli pair for active-qubit twirl.
    :param neighbor_P     : int 0-3 — T_p index for neighbor Pauli (0=I,1=X,2=Y,3=Z).
    :param neighbor_R     : int 0-2 — T_R index for neighbor rotation (0=Rx,1=Ry,2=Rz).
    :param coherent_noise : bool — if True and the norm of angles is >= 1e-8, use noisy_cnot(*angles) instead of ideal CNOT.
    :param angles         : list[float] or None — coherent-noise angles passed to noisy_cnot (required if coherent_noise).
    :param neighbor_noise : bool — if True, apply coherent noise on the neighbor qubit.
    :param neighbor_noise_angles : list[float] or None — (RX, RY, RZ) angles applied on the neighbor.
    :param turn_off_cRC   : bool — if True, skip the neighbor T / Ri twirl.
    """
    assert 0 <= neighbor_P <= 3, "neighbor_P must be in {0,1,2,3}"
    assert 0 <= neighbor_R <= 2, "neighbor_R must be in {0,1,2}"

    r   = QRoutine()
    qc  = r.new_wires(1)   # control
    qt  = r.new_wires(1)   # target
    qn  = r.new_wires(1)   # neighbor
 
    # ── Pre-CNOT region (left of first barrier) ──────────────────────────────
    # Neighbor: T  then  Ri(π/2)
    T_gate =  T_P_GATES[neighbor_P]
    Ri_fwd =  T_R_GATES[neighbor_R]
    Ri_inv = T_Rd_GATES[neighbor_R]

    if not turn_off_cRC:
        if T_gate is not None:
            r.apply(T_gate, qn)          # T  (Pauli from T_p)
        r.apply(Ri_fwd, qn)              # Ri(+π/2)  from T_R
 
 
    # ── CNOT region below ──────────────────────────────────────────────────────────
    # Active-qubit twirled CNOT (handles P, Q, R, S internally)
    # r.apply(rcCNOT(active_key,
    #                coherent_noise=coherent_noise,
    #                angles=angles),
    #         qc, qt)
    ckey = cnot_pauli_conj_table[active_key]
    
    key  = (int(active_key[0]), int(active_key[1]))
    ckey = (int(ckey[0]), int(ckey[1]))

    Paulis = [I, X, Y, Z]
    P_pre, Q_pre = Paulis[key[0]] , Paulis[key[1] ]
    P_pst, Q_pst = Paulis[ckey[0]], Paulis[ckey[1]]

    if P_pst is not I:
        r.apply(P_pst, qc)
    if Q_pst is not I:
        r.apply(Q_pst, qt)

    r.apply(barrier(), [qc, qt, qn])                      # ── first barrier ──

    if coherent_noise:
        assert angles is not None, "angles cannot be NONE for a noisy gate!!"
    if coherent_noise and np.linalg.norm(angles) >= 1e-8:
        r.apply(noisy_cnot(*angles), qc, qt)
    else:
        r.apply(CNOT, qc, qt)

    if neighbor_noise: 
        assert neighbor_noise_angles is not None
        r.apply(RX(neighbor_noise_angles[0]), qn)
        r.apply(RY(neighbor_noise_angles[1]), qn)
        r.apply(RZ(neighbor_noise_angles[2]), qn)

    r.apply(barrier(), [qc, qt, qn])                      # ── second barrier ──

    if P_pre is not I:
        r.apply(P_pre, qc)
    if Q_pre is not I:
        r.apply(Q_pre, qt)

    # ── CNOT region above ──────────────────────────────────────────────────────────


 
 
    # ── Post-CNOT region (right of second barrier) ───────────────────────────
    # Neighbor: Ri(-π/2)  then  T   (both are their own inverses / inverse)
    if not turn_off_cRC:    
        r.apply(Ri_inv, qn)              # Ri(-π/2)  =  Ri(+π/2)†
        if T_gate is not None:
            r.apply(T_gate, qn)          # T† = T  (Pauli self-inverse)
 
    return r