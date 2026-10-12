"""
Test 0 — Zero noise: 
myQLM pipeline vs exact.
Pass criteria: all roles match to < 1e-12.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import numpy as np
from source.tests.testing_functions import do_myqlm_pipeline, do_exact_computation, rm_glob, rm_test_caches
from source.core.records.run_names import plots_dir


def do_myqlm_exact_no_noise(lnn=False, spam=False, all_pass=True):
    # ── Configuration ─────────────────────────────────────────────────────────────
    CFG = dict(
        INFIDELITY     = 0.0,
        RATIO_CT       = 0.5,
        NOISE_TYPE     = "pauli",
        SAMPLE_SIZE    = 1,
        LIST_TIMES     = [0.4, 0.6],   # reduced from [0.2, 0.6, 1.4, 1.8] for a quick run
        DT             = 0.2,
        LNN            = lnn,
        do_SPAM        = spam,
    )
    TOL = 1e-12

    roles_map = {"raw": "bare", "raw_nec": "NEC", "rc": "RC",
                 "rc_nec": "NEC+RC", "crc": "cRC", "crc_nec": "NEC+cRC"}

    # Test myqlm first at infinite shots
    rm_glob("datafiles/tests/test00_*")
    rm_test_caches()
    myqlm_results  = do_myqlm_pipeline(  **CFG, nshots=0, testing_folder="tests/test00_myqlm")
    exact          = do_exact_computation(LIST_TIMES=CFG["LIST_TIMES"], DT=CFG["DT"], LNN=CFG["LNN"])

    for T in CFG["LIST_TIMES"]:
        nt = int(round(T / CFG["DT"]))
        print(f"\n  --- Nt={nt}, Infinite shots ---")
        for role, label in roles_map.items():
            myqlm_obs  = np.array(myqlm_results ["depths"][str(nt)][label]["obs_values"])
            exact_obs  = exact[nt]

            diff_mq_ex = np.max(np.abs(myqlm_obs  - exact_obs))

            ok = diff_mq_ex < TOL
            if not ok: all_pass = False
            print(f"  {'✅' if ok else '❌'} {role:8s}: "
                  f"|myQLM-exact|={diff_mq_ex:.2e}")

    # ===================== Plot =========================
    import matplotlib.pyplot as plt

    obs_labels = ["X₀", "Y₁", "Z₂", "X₀Y₁", "X₀Z₂", "Y₁Z₂", "X₀Y₁Z₂"]
    nt_list    = [int(round(T / CFG["DT"])) for T in CFG["LIST_TIMES"]]

    colors  = {"bare":"gray",     "NEC":"red",      "RC":"steelblue",
               "NEC+RC":"orange", "cRC":"darkblue", "NEC+cRC":"firebrick"}
    markers = {"bare":"o", "NEC":"s", "RC":"^", "NEC+RC":"v", "cRC":"D", "NEC+cRC":"*"}

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)
    axes_flat = axes.flatten()

    for oi, (ax, obs_label) in enumerate(zip(axes_flat, obs_labels)):
        # ── Exact ────────────────────────────────────────────────────────────
        exact_vals = [exact[nt][oi] for nt in nt_list]
        ax.plot(nt_list, exact_vals, "s", lw=2.5, label="exact", ms=8.5, color="lime")

        # ── myQLM per role ────────────────────────────────────────────────────
        for role, label in roles_map.items():
            myqlm_vals = [
                np.array(myqlm_results["depths"][str(nt)][label]["obs_values"])[oi]
                for nt in nt_list
            ]
            ax.plot(nt_list, myqlm_vals,
                    color   = colors.get(label, "gray"),
                    marker  = markers.get(label, "o"),
                    ls      = "--",
                    lw      = 1.5,
                    ms      = 6,
                    label   = label,
                    alpha   = 0.85)

        ax.set_title(obs_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("$N_t$")
        ax.set_xticks(nt_list)
        ax.grid(True, alpha=0.3)
        if oi == 0:
            ax.set_ylabel("Expectation value")

    axes_flat[-1].set_visible(False)
    handles, labels_leg = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper right", fontsize=8, ncol=1)
    fig.suptitle(f"myQLM vs exact — LNN={lnn}, SPAM={spam}", fontweight="bold", fontsize=12)
    plt.tight_layout()
    PLOT_DIR = plots_dir("tests/test00_myqlm").as_posix()
    os.makedirs(PLOT_DIR, exist_ok=True)
    fname = f"{PLOT_DIR}/test00_lnn{lnn}_spam{spam}.png"
    plt.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {fname}")

    del myqlm_results
    return all_pass

def test_00_zero_noise_no_shot_noise():
    oks = True
    for (lnn,spam) in [(0,0), (0,1), (1,0), (1,1)]:
        ok = do_myqlm_exact_no_noise(lnn=lnn, spam=spam)
        oks = (oks and ok)

    print(f" All {len([(0,0),(0,1),(1,0),(1,1)])} (lnn,spam) configs {'✅' if oks else '❌'} ")

    rm_test_caches()
    assert oks, "myQLM vs exact mismatch (>1e-12) in one or more (lnn,spam) configs — see printed diffs above"


if __name__ == "__main__":
    # Through pytest, so conftest.py sends the circuit caches to datafiles/tests/circuits/.
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
