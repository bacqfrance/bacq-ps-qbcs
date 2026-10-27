# Guide for adding a new dummy QPU

How to add a **dummy noisy emulator** of a QPU the benchmark was not run on to
`definitions_dummy_noisy_emulators.json`, and what to do when its pricing or timing needs a model the code does not have.
What a dummy is, the models and the JSON fields are in `USER_MANUAL §4.4`; the run names are in `USER_MANUAL §5.4`, the summary
fields in `USER_MANUAL §10.2`.

## 1. What a dummy is, and when to add one

A dummy is one local coherent-noise simulation (the same seed and path as `benchmark_coherent_noise_lnn.py` /
`benchmark_coherent_noise_all_to_all.py`) at the QPU's two-qubit gate infidelity, plus a modelled QPU time and cost from the QPU's
published prices and timings. Add one for a QPU you cannot run on (no access, or not yet) when you want its place in the
cost and time plots.

It can claim: a fidelity-level coherent noise, a QPU time and a cost estimated from the gate counts of the logical circuits.
It cannot claim: the QPU's real noise, its transpiled circuits, its queue (none is modelled), or a bill (the actual cost is 0;
the modelled cost is an estimate, labelled `[estimated]` in the collective plots).

## 2. Procedure A: a new dummy on the existing models

Edit only `definitions_dummy_noisy_emulators.json`, beside this guide. A new provider prefix also needs an entry in the
`QPU_PROVIDER` table of `source/analysis_scripts/bacq_ps_qbcs_summary_json.py`.

1. **The key `NAME`**: `<provider>-<backend>`, lower case (`ionq-forte`, `rigetti-cepheus-108q`). It becomes the device field of
   the run folders (`USER_MANUAL §5.4`: `…_sim-dummy_<NAME>_…`), so it must contain a `-` and no `_` (the part before the first `-` is the provider; run names are split on `_`).
   `NAME` may equal a real QPU's name; the data type `sim-dummy` tells the runs apart.
2. **The noise**: `architecture` (the real QPU's architecture as its manufacturer spells it, e.g. `Heron r2`; required, shown as
   `Architecture` in the summary), `Fidelity` (the two-qubit gate fidelity in %, from a data sheet or a vendor page; write the figure used and the
   version or date in `source_fidelity`), `r_ct` (crosstalk ratio; 0.0 for trapped ions), `noise_type` (`"Pauli"`) and `lnn`
   (`"True"` for a nearest-neighbour topology, `"False"` for all-to-all; the strings, not booleans).
3. **`pricing`**: choose a model and fill all of its constants:

   | `model` | Constants |
   |---|---|
   | `ibm_payg` | `usd_per_qpu_minute` |
   | `braket` | `usd_per_task`, `usd_per_shot`, `circuits_per_task` (100 for devices with program sets, else 1) |
   | `braket_scaled` | the same; the per-shot price is an estimate scaled from a similar machine |
   | `hqc` | `hqc_fixed`, `weights` (`n1q`, `n2q`, `nm`), `hqc_divisor`, `usd_per_hqc` |

4. **`timing`**: `t_1q`, `t_2q`, `t_meas` and `per_shot_overhead`, in seconds. Every cell must be filled: `null` (or a missing cell)
   is refused when the run starts and again by the analysis, never taken as 0.
5. **The three source fields** `source_fidelity`, `source_pricing`, `source_timing`: for each value its URL, quote or file and date.
   A value that was not found in a source is written as `ASSUMPTION`, with the rule used (for example the mean of other
   dummies' values), never invented silently. Give the lookup date of every price.
6. **Rules of thumb**: prefer data sheets and papers about the exact system; say which system a paper describes (a paper on a
   sibling machine is not the machine); when sources disagree, record all of them and say which one was used.
7. **Durations of IBM and IQM QPUs** can be taken from a calibration, as the median over the qubits and pairs of the gate
   (`sx`/`x` or `r` for `t_1q`; `cz` for `t_2q`) and measurement durations:
   - from a saved `calibration_snapshot.json` of a real run on that QPU (`USER_MANUAL §5.4`);
   - or from a read-only fetch of the backend's target (IBM: `connect_ibm(...)` then `service.backend("<name>").target`, in `venvs/ibm.venv`;
     IQM Resonance: `IQMProvider(...).get_backend(use_metrics=True)`, in `venvs/iqm_resonance.venv`, with the token of `tokens/README.md`).

   IBM's per-shot overhead can be the backend configuration's `default_rep_delay`. The values are copied into the JSON, so a run never
   needs live access.

Then run and check (section 4).

## 3. Procedure B: a new pricing or timing model

When no existing `pricing.model` fits (a different billing rule), or the time of a shot needs another formula:

1. **Code**: `source/core/analysis/dummy_estimates.py` holds the models. Add the model's constants to `PRICING_TERMS` (so a missing
   constant is refused), its branch to `modelled_cost()` (and to `modelled_qpu_time()` for a timing change), and keep the rule that
   a null term is an error.
2. **The summary**: the dummy branch of `source/analysis_scripts/bacq_ps_qbcs_summary_json.py` writes `cost_explanation`,
   `timing_notes` and `extra_info` from the run's `dummy_estimates.json`; the new model must be named there with its constants, its
   `source_pricing` and the date of the definitions, so the notes always match what the run was computed with.
3. **Tests**: add to `source/tests/test_30_dummy_estimates.py` a hand-computed example of the model (the numbers worked out in the
   test, not copied from the code) and a case for the refused null term; update the test table in `USER_MANUAL §9`.
4. **Docs**: the model tables of `USER_MANUAL §4.4` and of this guide.

## 4. Run and check

```bash
python3 source/benchmark_scripts/local_simulation_scripts/benchmark_dummy_noisy_emulators.py mini
python3 source/analysis_scripts/run_post_process.py mini --include-simulations-dummy
```

(from the project root, with `venvs/local_sim.venv` active: `source venvs/local_sim.venv/bin/activate`. The commands are written for Linux; in Windows Terminal activate with `.\venvs\local_sim.venv\Scripts\activate` and use `py` instead of `python3`.)

- The driver simulates every dummy of the JSON and skips the ones whose latest run is finished ("skipping"), so a new entry is
  the only one that runs; an unfinished run is resumed in its folder. A failed dummy prints a ❌ line on stderr and the exit
  code is 2.
- A finished run folder is `mini_sim-dummy_<NAME>_lnn-<bool>_service-local_spam-True_layout-none_<date>_#<N>` and holds
  `dummy_estimates.json`: the entry as the run used it, with no absolute path.
- Each noise setting (topology, noise type, infidelity, `r_ct`) has its own circuit cache (`USER_MANUAL §4.1`): a new `Fidelity`
  or `r_ct` builds a new cache, so build the largest sample size first.
- In `processed_results/summary.json` (copied to `analysis/plots+summaries/<run name>/`) look at `QPU_time_s` (the modelled
  time; `Queue_time_s` stays 0), `effective_cost_USD` / `effective_cost_EUR` (the estimate; `actual_cost_EUR` stays 0), `dummy_estimate`
  (the time, the cost and the gate counts it came from), `dummy_inputs`, and read `cost_explanation`, `timing_notes` and
  `extra_info`: they must state the model, the constants, the sources and the date of the definitions.
- In the collective plots (`USER_MANUAL §10`) the dummy's bars read `[estimated]`, and the run table gets an `estimated` column.

## 5. Checklist

- [ ] The JSON entry is complete (no `null`), the values are sourced, the assumptions are marked `ASSUMPTION` with their rule.
- [ ] `source_fidelity`, `source_pricing` and `source_timing` each carry URLs or files and dates.
- [ ] The `mini` run passes and a rerun prints "skipping".
- [ ] The summary notes (`cost_explanation`, `timing_notes`, `extra_info`) read correctly for the new dummy.
- [ ] A new model (procedure B) has its hand-computed test; the `USER_MANUAL` tables (§4.4, §9) are unchanged or updated.
- [ ] The tests pass (`pytest source/tests/test_30_dummy_estimates.py`, then the whole suite).
