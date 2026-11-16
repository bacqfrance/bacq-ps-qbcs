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
    """Z-weighted cumulative AWAE per depth along the last axis: entry k is sum(Z[:k+1]*awae[:k+1]) / sum(Z[:k+1])
    (NaN propagates). awae may have leading axes (e.g. one curve per bootstrap draw)."""
    Z = np.asarray(Z, dtype=float)
    awae = np.asarray(awae, dtype=float)
    return np.cumsum(Z * awae, axis=-1) / np.cumsum(Z)


def fe_quantities(bare_cum, nec_cum, rcnec_cum, crcnec_cum):
    """E_bare (Eq. 5.7), FE_coh_nec (Eq. 5.8) and FE_ct_nec (Eq. 5.9) from the cumulative AWAE of bare, NEC, NEC+RC
    and NEC+cRC, elementwise on arrays of any shape (one curve, or one per bootstrap draw). A NaN or zero NEC
    denominator counts as the 1.0 sentinel, like a NaN input."""
    bare_cum, nec_cum, rcnec_cum, crcnec_cum = (np.asarray(x, dtype=float) for x in (bare_cum, nec_cum, rcnec_cum, crcnec_cum))
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(nec_cum == 0, np.nan, nec_cum)
        coh  = rcnec_cum / safe
        ct   = np.maximum(rcnec_cum - crcnec_cum, 0.0) / safe
    return {"E_bare": np.fmin(bare_cum, 1.0), "FE_coh_nec": 1.0 - np.fmin(coh, 1.0), "FE_ct_nec": np.fmin(ct, 1.0)}


def best_schemes(summary):
    """The best mitigation schemes of a summary as [(tag, ems)]: ("N_max", best_ems_Nmax), then ("N_acc", best_ems_Nacc)
    and ("N_acc_strict", best_ems_Nacc_strict), each only when it is a scheme not listed yet. A None scheme is skipped
    (none beats 1), so the list is empty with no best scheme."""
    out = []
    for tag, key in (("N_max", "best_ems_Nmax"), ("N_acc", "best_ems_Nacc"), ("N_acc_strict", "best_ems_Nacc_strict")):
        ems = summary.get(key)
        if isinstance(ems, str) and all(ems != e for _, e in out):
            out.append((tag, ems))
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
                               T_queue_total=0.0, T_collect_total=None,
                               awae_cumul_ci95=None, per_depth_ci95=None, bootstrap_done=False):
    """Compute the derived quantities from BACQ D2.1.2 Eqs. 5.6-5.11.

    Uncertainties (USER_MANUAL, "NEC error mitigation and uncertainties"): awae_cumul_ci95[ems] is the list per depth
    of the bootstrap 95% interval [lo, hi] of that scheme's cumulative AWAE, per_depth_ci95[key] the same for
    E_bare / FE_coh_nec / FE_ct_nec (key with its "_spam" suffix). E_min and N_acc come from the full-run values;
    E_min_*_ci95 is the interval of the chosen scheme. N_acc_strict uses its own best scheme: at each depth the
    one with the smallest upper bound, and the largest depth with that bound below epsilon there and at every
    shallower depth. Without a bootstrap (bootstrap_done False) every interval and every *_strict field is None.

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
        # Eqs. 5.7-5.9 — E_bare(k), FE_coh and FE_crosstalk wrt NEC
        return {f"{k}{suffix}": v for k, v in fe_quantities(bare_cum, nec_cum, rcnec_cum, crcnec_cum).items()}

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

    # Uncertainties: the bootstrap interval of the chosen scheme, and N_acc_strict with its own best scheme
    def _ci(ems, i):
        if not bootstrap_done or ems is None or awae_cumul_ci95 is None:
            return None
        ci = awae_cumul_ci95.get(ems, [None] * n)[i]
        return None if ci is None else [float(ci[0]), float(ci[1])]

    strict = None
    if bootstrap_done and awae_cumul_ci95 is not None:
        upper = {ems: np.array([np.nan if c is None else c[1] for c in awae_cumul_ci95.get(ems, [None] * n)], dtype=float)
                 for ems in ems_list}
        ems_strict = [_best_ems(ems_list, [upper[ems][i] for ems in ems_list]) for i in range(n)]
        Nt_strict = 0
        for i in range(n):
            if ems_strict[i] is not None and upper[ems_strict[i]][i] < epsilon: Nt_strict = Nt_vals[i]
            else: break
        ind_strict = np.argmin(abs(Nt_vals-Nt_strict))   # N_acc_strict = 0: the depth-1 values, like N_acc
        best_strict = ems_strict[ind_strict]
        strict = {
            "N_acc_strict"          : Nt_strict,
            "E_min_Nacc_strict"     : _smin(cum[best_strict][ind_strict], 1.0) if best_strict else 1.0,
            "best_ems_Nacc_strict"  : best_strict,
            "N_cnot_acc_strict"     : cnots_per_trotter * Nt_strict if not lnn else ((cnots_per_trotter + 3) * Nt_strict + (Nt_strict % 2) * 3),
        }

    # Eq. 5.10 — timing fractions, with the queue as a third part of the total
    T_clas  = T_cpu_total;  T_quant = T_qpu_total;  T_queue = T_queue_total or 0.0
    T_collect = T_collect_total or 0.0
    T_total = T_clas + T_quant + T_queue + T_collect
    FT_clas  = _sdiv(T_clas,  T_total)
    FT_quant = _sdiv(T_quant, T_total)
    FT_queue = _sdiv(T_queue, T_total)
    FT_collect = _sdiv(T_collect, T_total) if T_collect_total is not None else "N/A"

    n_ems = len(base_ems_list)   # the 6 base schemes; the +SPAM variants reuse the same executed circuits

    result =  {
        "N_max"              : Nt_max,                                   # N^max_steps ,
        "E_min_Nmax"         : E_min[ind_Nt_max],                        # E_min at N^max
        "best_ems_Nmax"      : ems_with_Emin[ind_Nt_max],                # best ems at N^max (out of 6, or all 12 when SPAM data is available)
        "N_acc"              : Nt_acc,                                   # N^accurate_steps
        "E_min_Nacc"         : E_min[ind_Nt_acc],                        # E_min at N^acc
        "best_ems_Nacc"      : ems_with_Emin[ind_Nt_acc],                # best ems at N^acc
        "E_min_Nmax_ci95"    : _ci(ems_with_Emin[ind_Nt_max], ind_Nt_max),   # bootstrap 95% interval of the chosen scheme
        "E_min_Nacc_ci95"    : _ci(ems_with_Emin[ind_Nt_acc], ind_Nt_acc),
        "bootstrap_done"     : bool(bootstrap_done),
        "N_acc_strict"       : strict["N_acc_strict"] if strict else None,          # from the upper 95% bounds
        "E_min_Nacc_strict"  : strict["E_min_Nacc_strict"] if strict else None,
        "E_min_Nacc_strict_ci95": _ci(strict["best_ems_Nacc_strict"], ind_strict) if strict else None,
        "best_ems_Nacc_strict": strict["best_ems_Nacc_strict"] if strict else None,
        "N_cnot_max"         : N_cnot_max,        # N^max_2q
        "N_cnot_acc"         : N_cnot_acc,            # N^acc_2q
        "N_cnot_acc_strict"  : strict["N_cnot_acc_strict"] if strict else None,
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
        result[f"{key}_Nacc_strict"] = arr[ind_strict] if strict else None
        cis = (per_depth_ci95 or {}).get(key) if bootstrap_done else None
        for tag, i in (("Nmax", ind_Nt_max), ("Nacc", ind_Nt_acc), ("Nacc_strict", ind_strict if strict else None)):
            ci = None if (cis is None or i is None) else cis[i]
            result[f"{key}_{tag}_ci95"] = None if ci is None else [float(ci[0]), float(ci[1])]

    return result

# ── Round float values to 4 decimal places ───────────────────────────────────
def round_floats(obj, decimals=4):
    if isinstance(obj, float): return round(obj, decimals)
    if isinstance(obj, dict):  return {k: round_floats(v, decimals) for k, v in obj.items()}
    return obj
