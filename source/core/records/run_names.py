"""
run_names.py
The one place run-folder, plot-folder and summary names are built and parsed.

    <benchmark>_<data type>_<provider>-<backend>_lnn-<bool>_service-<svc>_spam-<bool>_layout-<mode>

Fields are separated by "_"; inside a field "-" joins a name and its value and "+" joins the
parameters of <benchmark>:

    <benchmark>   full | mini | full+nshots-10 | custom+ndepths-3+nsamples-8+nshots-50
                  | custom+nt-1-5-9+nsamples-8+nshots-50 | health-check+reps-1+nshots-100
                  (a subset analysis of N samples appends +nsamples-N, or replaces it)
    <data type>   real-data | sim-noiseless | sim-coherent | sim-fake-fez | sim-qpu-noise
    <provider>-<backend>   ibm-fez, iqm-emerald, iqm-vlq, iqm-eqe1, aqt-piast-q; local-aer for
                  simulations without a QPU's noise. A sim-qpu-noise run names the QPU it simulates.
    lnn / spam    True | False | none (standalone health checks)
    service       ibm-cloud | iqm-resonance | qaas-it4i | mqss-lrz | pcss | local
    layout        optimized | random | lowest | legacy | none | three qubits (15-14-21)

e.g. full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized.
Collective plots: <benchmark>_collective_<data type>[+<data type>...]; summaries: processed_results/summary.json
(summary+nsamples-N.json) in the run, copied to analysis/summaries/summary_<run name>.json.
Every other subset-analysis output in processed_results/ carries the same subset_suffix(N) = +nsamples-N
(run_data+nsamples-N.json, run_data_spam+nsamples-N.json, health_checks+nsamples-N.json, spam_confusion_matrix+nsamples-N.npy, plots).
The test suite's runs live in datafiles/tests/ (run folder tests/<run name>); their plots and summaries
go to analysis/tests/plots/ and analysis/tests/summaries/ (plots_dir(), summaries_dir()).
A new QPU provider or data type needs its value added here (QPUS / DATA_TYPES) and in test_20.
"""

import re
from dataclasses import dataclass, replace
from pathlib import Path

DATA_TYPES = ("real-data", "sim-noiseless", "sim-coherent", "sim-fake-fez", "sim-qpu-noise")

# The QPU script folders (source/benchmark_scripts/real_qpu_scripts/<QPU>/) → (provider, service, service name).
# The service is the access platform, as the run-name field (lower case, "-" — "_" separates the fields);
# the service name is the same platform as the code and the saved records call it (service_name=…).
QPUS = {
    "IBM":           ("ibm", "ibm-cloud",     "IBM_Cloud"),
    "IQM_RESONANCE": ("iqm", "iqm-resonance", "IQM_Resonance"),
    "IQM_VLQ":       ("iqm", "qaas-it4i",     "QAAS_IT4I"),
    "IQM_EQE1":      ("iqm", "mqss-lrz",      "MQSS_LRZ"),
    "AQT_PIAST-Q":   ("aqt", "pcss",          "PCSS"),
}
# The service names as constants, for the code that dispatches on them (qbcs, health_check, job_timing, ...):
# a typo then fails at import, and the strings live only in QPUS.
IBM_CLOUD     = QPUS["IBM"][2]
IQM_RESONANCE = QPUS["IQM_RESONANCE"][2]
QAAS_IT4I     = QPUS["IQM_VLQ"][2]
MQSS_LRZ      = QPUS["IQM_EQE1"][2]
PCSS          = QPUS["AQT_PIAST-Q"][2]
SERVICES      = tuple(q[1] for q in QPUS.values()) + ("local",)


def service_name(qpu):
    """The service name (service_name=…, saved in every record) of the QPU script folder `qpu`."""
    return QPUS[qpu][2]


def service_name_of_run_service(service):
    """The service name of a run name's service field (ibm-cloud → IBM_Cloud; local → local)."""
    return {q[1]: q[2] for q in QPUS.values()}.get(service, service if service == "local" else None)

_LAYOUT = re.compile(r"optimized|random|lowest|legacy|none|\d+-\d+-\d+")
_BENCH  = re.compile(r"[a-z0-9][a-z0-9-]*(\+[a-z]+-[0-9][0-9-]*)*")


@dataclass(frozen=True)
class RunName:
    benchmark: str
    data_type: str
    provider:  str
    backend:   str
    lnn:       str      # "True" | "False" | "none"
    service:   str
    spam:      str
    layout:    str

    def __str__(self):
        return (f"{self.benchmark}_{self.data_type}_{self.provider}-{self.backend}_lnn-{self.lnn}"
                f"_service-{self.service}_spam-{self.spam}_layout-{self.layout}")

    @property
    def device(self):
        """<provider>-<backend>, e.g. ibm-fez — what the summary and the plots call the run's QPU."""
        return f"{self.provider}-{self.backend}"

    @property
    def is_simulation(self):
        return self.data_type != "real-data"

    @property
    def figure_label(self):
        """What the figures call the run: the QPU (or, for sim-fake-fez, fake-fez) and whether it is the
        real QPU or a simulation, e.g. "ibm-fez — real QPU", "aqt-piast-q — simulation (calibration noise)"."""
        name = "fake-fez" if self.data_type == "sim-fake-fez" else self.device
        kind = {"real-data":     "real QPU",
                "sim-noiseless": "noiseless simulation",
                "sim-coherent":  "simulation (coherent CNOT noise)",
                "sim-fake-fez":  "simulation (FakeFez noise)",
                "sim-qpu-noise": "simulation (calibration noise)"}[self.data_type]
        return f"{name} — {kind}"

    @property
    def is_health_check(self):
        return self.benchmark.startswith("health-check")

    def with_samples(self, n):
        """The name of an analysis of the run's first n samples: +nsamples-n in <benchmark> (replacing
        a custom benchmark's own nsamples), for its subset summary and plots."""
        if n is None:
            return self
        parts = [p for p in self.benchmark.split("+") if not p.startswith("nsamples-")]
        return replace(self, benchmark="+".join(parts + [f"nsamples-{n}"]))


def benchmark_field(preset=None, nshots=None, n_samples=None, n_depths=None, nt_list=None):
    """<benchmark> from the benchmark arguments (source/core/runner_support/benchmark_args.py): a preset (full, mini),
    with +nshots-N when its shot count was changed, or a custom benchmark spelled out."""
    if preset:
        return preset if nshots is None else f"{preset}+nshots-{nshots}"
    depths = f"ndepths-{n_depths}" if nt_list is None else "nt-" + "-".join(map(str, nt_list))
    return f"custom+{depths}+nsamples-{n_samples}+nshots-{nshots}"


def health_check_benchmark(reps, nshots):
    return f"health-check+reps-{reps}+nshots-{nshots}"


def _value(text):
    """A field value: lower case, "_" and spaces become "-" (names keep "_" for separating fields)."""
    return re.sub(r"[_\s]+", "-", str(text).strip()).lower()


def _flag(b):
    return "none" if b is None else str(bool(b))


def qpu_device(qpu, backend):
    """(provider, backend) of a QPU script folder's backend= value: IBM ibm_fez → (ibm, fez),
    IQM_RESONANCE emerald → (iqm, emerald), IQM_VLQ iqm_vlq / VLQ → (iqm, vlq), IQM_EQE1 EQE1 → (iqm, eqe1),
    AQT_PIAST-Q PIAST-Q → (aqt, piast-q)."""
    provider = QPUS[qpu][0]
    b = _value(backend)
    for prefix in (f"{provider}-", "resonance-"):
        b = b.removeprefix(prefix)
    return provider, b


def qpu_run_name(qpu, backend, benchmark, lnn, do_spam, layout_tag, data_type="real-data"):
    """The run folder of a real-QPU script (data_type real-data) or of its simulate script
    (sim-qpu-noise, service local). layout_tag: LayoutArgs.tag (layout-optimized, …) or the bare mode."""
    provider, b = qpu_device(qpu, backend)
    service = QPUS[qpu][1] if data_type == "real-data" else "local"
    return str(RunName(benchmark, data_type, provider, b, _flag(lnn), service, _flag(do_spam),
                       layout_tag.removeprefix("layout-")))


def layout_field(mode, qubits=None):
    """The layout field of a run name: layout-optimized, layout-random, layout-lowest, layout-15-14-21 (mode "list")."""
    return "layout-" + ("-".join(map(str, qubits)) if mode == "list" else mode)


def run_name_with_layout(prefix, layout_tag):
    """qpu_run_prefix() + its layout field (layout_field() / LayoutArgs.tag)."""
    return f"{prefix}_{layout_tag}"


def run_name_with_any_layout(prefix):
    """Glob pattern for every run folder named qpu_run_prefix() + _layout-<any>."""
    return f"{prefix}_layout-*"


def qpu_run_prefix(qpu, backend, benchmark, lnn, do_spam, data_type="real-data"):
    """qpu_run_name() up to its layout field: what the collect / job-status scripts look for
    (benchmark_args.find_collect_run_folder() adds _layout-<mode>)."""
    return qpu_run_name(qpu, backend, benchmark, lnn, do_spam, "none", data_type).removesuffix("_layout-none")


def health_check_run_name(qpu, backend, reps, nshots):
    """The folder of a standalone health check (health_check_real_*.py)."""
    provider, b = qpu_device(qpu, backend)
    return str(RunName(health_check_benchmark(reps, nshots), "real-data", provider, b, "none",
                       QPUS[qpu][1], "none", "none"))


def local_run_name(benchmark, data_type, lnn, do_spam=True, device="local-aer"):
    """The run folder of a local-simulation script (no QPU, layout none)."""
    provider, b = device.split("-", 1)
    return str(RunName(benchmark, data_type, provider, b, _flag(lnn), "local", _flag(do_spam), "none"))


def parse_run_name(name):
    """RunName of a run-folder name (or a path ending in one); ValueError if it doesn't follow the scheme."""
    base   = str(name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    fields = base.split("_")
    if len(fields) != 7:
        raise ValueError(f"{base!r} is not a run name: expected 7 fields separated by '_', got {len(fields)}")
    bench, dtype, device, lnn, service, spam, layout = fields[0], fields[1], fields[2], *fields[3:]
    def tail(field, key):
        if not field.startswith(key + "-"):
            raise ValueError(f"{base!r}: expected a {key}-… field, got {field!r}")
        return field[len(key) + 1:]
    lnn, service, spam, layout = tail(lnn, "lnn"), tail(service, "service"), tail(spam, "spam"), tail(layout, "layout")
    if not _BENCH.fullmatch(bench):
        raise ValueError(f"{base!r}: bad benchmark field {bench!r}")
    if dtype not in DATA_TYPES:
        raise ValueError(f"{base!r}: unknown data type {dtype!r} (one of {', '.join(DATA_TYPES)})")
    if "-" not in device:
        raise ValueError(f"{base!r}: expected <provider>-<backend>, got {device!r}")
    provider, backend = device.split("-", 1)
    if lnn not in ("True", "False", "none") or spam not in ("True", "False", "none"):
        raise ValueError(f"{base!r}: lnn and spam must be True, False or none")
    if service not in SERVICES:
        raise ValueError(f"{base!r}: unknown service {service!r} (one of {', '.join(SERVICES)})")
    if not _LAYOUT.fullmatch(layout):
        raise ValueError(f"{base!r}: bad layout {layout!r} (optimized, random, lowest, legacy, none or three qubits a-b-c)")
    return RunName(bench, dtype, provider, backend, lnn, service, spam, layout)


def entry_data_type(entry):
    """The data type of a run name without its <benchmark> field (a RUN_THESE entry of run_post_process.py,
    "<data type>_<provider>-<backend>…", possibly cut short); ValueError for an unknown one."""
    dtype = str(entry).split("_", 1)[0]
    if dtype not in DATA_TYPES:
        raise ValueError(f"{entry!r}: unknown data type {dtype!r} (one of {', '.join(DATA_TYPES)})")
    return dtype


def is_run_name(name):
    try:
        parse_run_name(name)
        return True
    except ValueError:
        return False


def subset_suffix(n_samples=None):
    """+nsamples-N, the suffix of every subset-analysis output (run_data+nsamples-N.json, ...); "" for the full run (None)."""
    return "" if n_samples is None else f"+nsamples-{n_samples}"


def run_summary_name(n_samples=None):
    """The summary json in the run's own processed_results/: summary.json, or summary+nsamples-N.json for an
    analysis of the run's first N samples. Short, because the folder already carries the run name (and
    datafiles/<run name>/processed_results/summary_<run name>.json would pass Windows' 260-character limit)."""
    return f"summary{subset_suffix(n_samples)}.json"


def summary_name(run_folder, n_samples=None):
    """summary_<run name>.json — the copy in analysis/summaries/, which collects every run's summary.
    n_samples: a subset analysis of the run's first n samples (+nsamples-n in <benchmark>)."""
    return f"summary_{parse_run_name(run_folder).with_samples(n_samples)}.json"


TESTS_FOLDER = "tests"   # datafiles/tests/ holds the test suite's runs and caches; analysis/tests/ its plots and summaries


def is_test_run(run_folder):
    """True for a run folder of the test suite: tests/<run name> (relative to datafiles/)."""
    return str(run_folder).replace("\\", "/").split("/", 1)[0] == TESTS_FOLDER


def _analysis_root(run_folder):
    return Path("analysis", TESTS_FOLDER) if is_test_run(run_folder) else Path("analysis")


def plots_dir(run_folder):
    """A run's plot folder, relative to the project root: analysis/plots/<run folder> (old_runs_full/… keeps its
    subfolder), or analysis/tests/plots/<run name> for a test run."""
    rel = str(run_folder).replace("\\", "/")
    if is_test_run(rel):
        rel = rel.split("/", 1)[1]
    return _analysis_root(run_folder) / "plots" / rel


def summaries_dir(run_folder=None):
    """Where run_post_process.py collects the summary jsons: analysis/summaries, or analysis/tests/summaries for a
    test run."""
    return _analysis_root(run_folder or "") / "summaries"


def collective_plots_dir(name, test=False):
    """A collective plot folder (collective_folder_name()), under analysis/plots/ or, for test runs, analysis/tests/plots/."""
    return Path("analysis", *([TESTS_FOLDER] if test else []), "plots", name)


def collective_folder_name(benchmark, data_types):
    """<benchmark>_collective_<data type>[+<data type>...] for the plots comparing several runs."""
    return f"{benchmark}_collective_{'+'.join(data_types)}"
