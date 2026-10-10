# Helper functions    
from qat.lang.AQASM import CNOT, RX, RZ, H
import numpy as np
from source.core.circuits.rc_functions import rcCNOT


def XX_gate(qroutine, qreg, theta, q0, q1):
    qroutine.apply(H, qreg[q0])
    qroutine.apply(H, qreg[q1])
    qroutine.apply(CNOT, [qreg[q0], qreg[q1]])
    qroutine.apply(RZ(theta), qreg[q1])
    qroutine.apply(CNOT, [qreg[q0], qreg[q1]])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(H, qreg[q1])

def YY_gate(qroutine, qreg, theta, q0, q1):
    qroutine.apply(RX(np.pi/2), qreg[q0])
    qroutine.apply(RX(np.pi/2), qreg[q1])
    qroutine.apply(CNOT, [qreg[q0], qreg[q1]])
    qroutine.apply(RZ(theta), qreg[q1])
    qroutine.apply(CNOT, [qreg[q0], qreg[q1]])
    qroutine.apply(RX(-np.pi/2), qreg[q0])
    qroutine.apply(RX(-np.pi/2), qreg[q1])

def rot_XY(qroutine, qreg, theta, q0, q1):
    qroutine.apply(RZ(-np.pi/2), qreg[q0])
    qroutine.apply(RZ(-np.pi/2), qreg[q1])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(H, qreg[q1])
    qroutine.apply(RZ(np.pi/2), qreg[q0])
    qroutine.apply(RZ(-np.pi/2), qreg[q1])
    qroutine.apply(CNOT, qreg[q0], qreg[q1])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(RZ(np.pi-theta), qreg[q0])
    qroutine.apply(RZ(theta), qreg[q1])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(CNOT, qreg[q0], qreg[q1])
    qroutine.apply(RZ(np.pi/2), qreg[q0])
    qroutine.apply(RZ(-np.pi/2), qreg[q1])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(H, qreg[q1])
    qroutine.apply(RZ(-np.pi/2), qreg[q0])
    qroutine.apply(RZ(-np.pi/2), qreg[q1])

def rot_XY_RC(qroutine, qreg, theta, q0, q1, PQ_key, coherent_noise=None):
    '''
    XY rotation by theta on qubits q0, q1, built from two Pauli-twirled CNOTs (rcCNOT) keyed by PQ_key.
    
    :param qroutine: QRoutine()
    :param qreg: register
    :param theta: rotation angle
    :param q0: qubit 1
    :param q1: qubit 2
    :param PQ_key: list[str] -> ["ij", "kl"] ; 0 <= i,j,k,l <= 3
    '''
    if coherent_noise is not None:
        CNOT_noise = True
    else:
        CNOT_noise = False
    assert len(PQ_key)==2, "The key for twirling should be a list with two elements of the form 'ij'"
    qroutine.apply(RZ(-np.pi/2), qreg[q0])
    qroutine.apply(RZ(-np.pi/2), qreg[q1])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(H, qreg[q1])
    qroutine.apply(RZ(np.pi/2), qreg[q0])
    qroutine.apply(RZ(-np.pi/2), qreg[q1])
    qroutine.apply(rcCNOT(PQ_key[0], coherent_noise=CNOT_noise, angles=coherent_noise), qreg[q0], qreg[q1])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(RZ(np.pi-theta), qreg[q0])
    qroutine.apply(RZ(theta), qreg[q1])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(rcCNOT(PQ_key[1], coherent_noise=CNOT_noise, angles=coherent_noise), qreg[q0], qreg[q1])
    qroutine.apply(RZ(np.pi/2), qreg[q0])
    qroutine.apply(RZ(-np.pi/2), qreg[q1])
    qroutine.apply(H, qreg[q0])
    qroutine.apply(H, qreg[q1])
    qroutine.apply(RZ(-np.pi/2), qreg[q0])
    qroutine.apply(RZ(-np.pi/2), qreg[q1])





