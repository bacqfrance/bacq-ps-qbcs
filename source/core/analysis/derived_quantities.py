"""
derived_quantities.py
The quantities a run summary derives from the AWAE and the times (accuracy reached per depth,
FT_* time fractions, per-EMS QPU times, best mitigation), and the JSON helpers the summary uses (NaNEncoder, round_floats).
"""

import json
import math

import numpy as np

from source.core.analysis.mitigation import BASE_EMS_LIST

EPSILON = 0.05   # accuracy threshold for N_acc


# ========= Analysis functions ==================================

def cumulative_awae(Z, awae):
    """Z-weighted cumulative AWAE per depth: entry k is sum(Z[:k+1]*awae[:k+1]) / sum(Z[:k+1]) (NaN propagates).

    The order of operations (slice sums, then the product with 1/sum(Z)) is fixed so that summary.json
    stays bit-identical; do not replace it with a cumsum.
    """
    Z = np.asarray(Z, dtype=float)
    awae = np.asarray(awae, dtype=float)
    return np.array([(1 / np.sum(Z[:k+1])) * np.sum(Z[:k+1] * awae[:k+1]) for k in range(len(Z))], dtype=float)


def best_schemes(summary):
    """The best mitigation schemes of a summary as [(tag, ems)]: ("N_max", best_ems_Nmax), then ("N_acc", best_ems_Nacc)
    when it is a different scheme. A None scheme is skipped (none beats 1), so the list is empty with no best scheme;
    legacy summaries without best_ems_Nacc give only N_max."""
    best_max = summary.get("best_ems_Nmax")
    best_acc = summary.get("best_ems_Nacc")
    out = [("N_max", best_max)] if best_max is not None else []
    if isinstance(best_acc, str) and best_acc != best_max:
        out.append(("N_acc", best_acc))
    return out


class NaNEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

    def iterencode(self, o, _one_shot=False):
        for chunk in super().iterencode(o, _one_shot):
            yield chunk.replace("NaN", "null")

def compute_derived_quantities(awae_cumul_Nt, awae_Nt, T_cpu_total, T_qpu_total,
                               epsilon=EPSILON, cnots_per_trotter=6, Nt_vals=None, analyze_only_bare=False, lnn=True,
                               T_queue_total=0.0, T_collect_total=None):
    """Compute the derived quantities from BACQ D2.1.2 Eqs. 5.6-5.11.

    awae_cumul_Nt / awae_Nt may carry just the 6 base schemes
    (bare, NEC, RC, NEC+RC, cRC, NEC+cRC) or, when SPAM-corrected data is
    also available, all 12 (the 6 base schemes plus their "<scheme>+SPAM"
    counterparts). E_min/best_ems is always the argmin over whatever set of
    schemes is present (6 or 12) — i.e. it picks the best mitigation
    including whether SPAM correction helps; it is None at a depth where no scheme beats 1
    (all NaN, or none below 1). E_bare/FE_coh_nec/FE_ct_nec are
    computed once per SPAM-state present (suffixed "_spam" for the SPAM one).

    Timing: T_total = T_cpu + T_qpu + T_queue + T_collect, and
    FT_clas + FT_quant + FT_queue + FT_collect = 1 (timing_definitions_version 3; version 2
    had no collection term, before it no queue term). T_collect None (runs collected before
    the collection was timed) is reported as "N/A" and counts as 0.
    """

    assert Nt_vals is not None, "The list of circuit depths cannot be empty!!"
    Nt_vals = np.array(Nt_vals)
    assert np.prod(Nt_vals) > 0, "Nt_vals cannot have 0s or negative integers!!"
    Nt_max = np.max(Nt_vals)
    N_cnot_max = cnots_per_trotter * Nt_max if not lnn else ((cnots_per_trotter + 3) * Nt_max + (Nt_max%2)*3)
    ind_Nt_max = np.argmin(abs(Nt_vals-Nt_max))

    def _smin(*args):
        vals = [v for v in args if not (isinstance(v, float) and math.isnan(v))]
        return min(vals) if vals else float("nan")

    def _best_ems(ems_names, vals):
        # the scheme with the smallest value, or None when none beats the 1.0 sentinel (all NaN, or none below 1)
        arr = np.array(vals, dtype=float)
        if np.all(np.isnan(arr)) or np.nanmin(arr) >= 1:
            return None
        return ems_names[int(np.nanargmin(arr))]

    def _sdiv(a, b):
        return float("nan") if (b == 0 or (isinstance(b, float) and math.isnan(b))) else a / b

    base_ems_list = BASE_EMS_LIST
    has_spam = all(f"{ems}+SPAM" in awae_cumul_Nt for ems in base_ems_list)
    ems_list = base_ems_list + [f"{ems}+SPAM" for ems in base_ems_list] if has_spam else list(base_ems_list)

    n = len(awae_cumul_Nt[base_ems_list[0]])
    assert len(Nt_vals)==n

    cum = {ems: np.array(awae_cumul_Nt[ems], dtype=float) for ems in ems_list}

    # Eq. 5.6 — E_min(k), argmin over every scheme present (6 base, or all 12 when SPAM data is available)
    if not analyze_only_bare:
        E_min         = np.array([_smin(*(cum[ems][i] for ems in ems_list), 1.0) for i in range(n)])
        ems_with_Emin = [_best_ems(ems_list, [cum[ems][i] for ems in ems_list]) for i in range(n)]
    else:
        E_min         = np.array([_smin(cum["bare"][i], 1.0) for i in range(n)])
        ems_with_Emin = ["bare"]*n

    def _spam_block(suffix, bare_cum, nec_cum, rcnec_cum, crcnec_cum):
        # Eq. 5.7 — E_bare(k)
        E_bare = np.array([_smin(bare_cum[i], 1.0) for i in range(n)])
        # Eq. 5.8 — FE_coh wrt NEC
        FE_coh_nec = np.array([1.0 - _smin(_sdiv(rcnec_cum[i], nec_cum[i]), 1.0) for i in range(n)])
        # Eq. 5.9 — FE_crosstalk wrt NEC
        FE_ct_nec = np.array([_smin(_sdiv(max(rcnec_cum[i]-crcnec_cum[i], 0.), nec_cum[i]), 1.0) for i in range(n)])
        return {
            f"E_bare{suffix}": E_bare,
            f"FE_coh_nec{suffix}": FE_coh_nec,
            f"FE_ct_nec{suffix}": FE_ct_nec,
        }

    per_depth = _spam_block("", cum["bare"], cum["NEC"], cum["NEC+RC"], cum["NEC+cRC"])
    if has_spam:
        per_depth.update(_spam_block("_spam", cum["bare+SPAM"], cum["NEC+SPAM"], cum["NEC+RC+SPAM"], cum["NEC+cRC+SPAM"]))

    # Qty 8 — per-depth E_min < epsilon, consecutive from k=1
    Nt_acc = 0
    for i in range(n):
        if not math.isnan(E_min[i]) and E_min[i] < epsilon: Nt_acc = Nt_vals[i]
        else: break

    N_cnot_acc = cnots_per_trotter * Nt_acc if not lnn else ((cnots_per_trotter + 3) * Nt_acc + (Nt_acc % 2) * 3)
    ind_Nt_acc = np.argmin(abs(Nt_vals-Nt_acc))

    # Eq. 5.10 — timing fractions, with the queue as a third part of the total
    T_clas  = T_cpu_total;  T_quant = T_qpu_total;  T_queue = T_queue_total or 0.0
    T_collect = T_collect_total or 0.0
    T_total = T_clas + T_quant + T_queue + T_collect
    FT_clas  = _sdiv(T_clas,  T_total)
    FT_quant = _sdiv(T_quant, T_total)
    FT_queue = _sdiv(T_queue, T_total)
    FT_collect = _sdiv(T_collect, T_total) if T_collect_total is not None else "N/A"

    n_ems = len(base_ems_list) * (2 if has_spam else 1)

    result =  {
        "N_max"              : Nt_max,                                   # N^max_steps ,
        "E_min_Nmax"         : E_min[ind_Nt_max],                        # E_min at N^max
        "best_ems_Nmax"      : ems_with_Emin[ind_Nt_max],                # best ems at N^max (out of 6, or all 12 when SPAM data is available)
        "N_acc"              : Nt_acc,                                   # N^accurate_steps
        "E_min_Nacc"         : E_min[ind_Nt_acc],                        # E_min at N^acc
        "best_ems_Nacc"      : ems_with_Emin[ind_Nt_acc],                # best ems at N^acc
        "N_cnot_max"         : N_cnot_max,        # N^max_2q
        "N_cnot_acc"         : N_cnot_acc,            # N^acc_2q
        "QPU_time_s"         : T_quant,
        "CPU_time_s"         : T_clas,
        "Queue_time_s"       : T_queue,
        "QPU_time_per_EMS_s" : T_quant/n_ems,
        "CPU_time_per_EMS_s" : T_clas/n_ems,
        "Queue_time_per_EMS_s": T_queue/n_ems,
        "Collect_time_s"     : T_collect_total if T_collect_total is not None else "N/A",
        "FT_clas"            : FT_clas,                          # FT_clas
        "FT_quant"           : FT_quant,                         # FT_quant
        "FT_queue"           : FT_queue,                         # FT_queue
        "FT_collect"         : FT_collect,                       # result collection
        "T_generation_submission_execution_s"       : T_total,                           # D_exec (CPU + QPU + queue + collection)
    }
    for key, arr in per_depth.items():
        result[f"{key}_Nmax"] = arr[ind_Nt_max]
        result[f"{key}_Nacc"] = arr[ind_Nt_acc]

    return result

# ── Round float values to 4 decimal places ───────────────────────────────────
def round_floats(obj, decimals=4):
    if isinstance(obj, float): return round(obj, decimals)
    if isinstance(obj, dict):  return {k: round_floats(v, decimals) for k, v in obj.items()}
    return obj
