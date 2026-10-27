"""
run_names.py
The one place run-folder, plot-folder and summary names are built and parsed.

    <benchmark>_<data type>_<provider>-<backend>_lnn-<bool>_service-<svc>_spam-<bool>_layout-<mode>_<YYYY-MM-DD>_#<N>

Fields are separated by "_"; inside a field "-" joins a name and its value and "+" joins the
parameters of <benchmark>:

    <benchmark>   full | mini | full+nshots-10 | custom+ndepths-3+nsamples-8+nshots-50
                  | custom+nt-1-5-9+nsamples-8+nshots-50 | health-check+reps-1+nshots-100
                  (a subset analysis of N samples appends +nsamples-N, or replaces it)
    <data type>   real-data | sim-noiseless | sim-coherent | sim-dummy | sim-fake-fez | sim-qpu-noise
    <provider>-<backend>   ibm-fez, iqm-emerald, iqm-vlq, iqm-eqe1, aqt-piast-q; local-aer for
                  simulations without a QPU's noise. A sim-qpu-noise run names the QPU it simulates, a sim-dummy
                  run the QPU it is a dummy emulator of (ibm-marrakesh, ionq-forte, ...: a real run of that device
                  has data type real-data, so a device alone does not tell the two apart).
    lnn / spam    True | False | none (standalone health checks)
    service       ibm-cloud | iqm-resonance | qaas-it4i | mqss-lrz | pcss | local
    layout        optimized | random | lowest | legacy | none | three qubits (15-14-21)
    date          the UTC day the run folder was first made by its submit (simulate, health-check) script
    #N            the run's number that day: #1 for the first run of those settings, #2 for the next, ...

e.g. full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized_2026-10-11_#1.
The builders below (qpu_run_name(), local_run_name(), health_check_run_name()) give the name without its date;
dated_run_folder() adds it and the number: it reuses the latest folder of those settings while it is unfinished, else
makes a new one dated today with the next number. Names without a date (test runs) still parse, with date None.
Collective plots: <benchmark>_collective_<data type>[+<data type>...][_dates-…|_date-… for a single day][_backends-…][_services-…][_layouts-…][_latest-per-qpu|_best-nacc-per-qpu|_best-emin-at-nmax-per-qpu]
(the optional parts name run_post_process.py's filters, RunFilter.suffix(), and --latest-per-qpu / --best-Nacc-per-qpu / --best-Emin-at-Nmax-per-qpu); summaries: processed_results/summary.json
(summary+nsamples-N.json) in the run, copied to analysis/plots+summaries/<run folder>/summary_<run name>.json.
Every other subset-analysis output in processed_results/ carries the same subset_suffix(N) = +nsamples-N
(run_data+nsamples-N.json, run_data_spam+nsamples-N.json, health_checks+nsamples-N.json, spam_confusion_matrix+nsamples-N.npy, plots).
The test suite's runs live in datafiles/tests/ (run folder tests/<run name>); their plots and summaries
go to analysis/tests/plots+summaries/ (plots_dir()).
A new QPU provider or data type needs its value added here (QPUS / DATA_TYPES) and in test_20; a data type also needs its figure_label,
and its metadata goes in bacq_ps_qbcs_summary_json.py.
"""

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

DATA_TYPES = ("real-data", "sim-noiseless", "sim-coherent", "sim-dummy", "sim-fake-fez", "sim-qpu-noise")

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

# Display labels of a layout (layout_label()): not part of any run name.
LAYOUT_BY_GATES        = "optimized_by_gates"                       # optimized before T1/T2 entered the layout score
LAYOUT_EMERALD_LEGACY  = "optimized_per_chunk_by_2q_gates_IQM"      # the legacy layout of iqm-emerald
LAYOUT_LABELS          = (LAYOUT_BY_GATES, LAYOUT_EMERALD_LEGACY)
LAST_DATE_WITHOUT_T1T2 = "2026-10-19"   # runs of this day or earlier are "by gates" (T1/T2 came in with the commit of 2026-10-19 23:24)
_DATAFILES = Path(__file__).resolve().parents[3] / "datafiles"


def layout_label(run):
    """How the figures and tables call the layout of `run` (a run folder, relative to datafiles/, or a RunName):
    the layout field, except that
    - "optimized" becomes optimized_by_gates for a run dated LAST_DATE_WITHOUT_T1T2 or earlier, or whose
      transpiled_circuits/physical_qubits.json has no "decoherence_included" key (the layout score had no T1/T2 term yet);
    - "legacy" on iqm-emerald becomes optimized_per_chunk_by_2q_gates_IQM."""
    name = run if isinstance(run, RunName) else parse_run_name(run)
    if name.layout == "legacy" and name.device == "iqm-emerald":
        return LAYOUT_EMERALD_LEGACY
    if name.layout != "optimized":
        return name.layout
    if name.date and name.date <= LAST_DATE_WITHOUT_T1T2:
        return LAYOUT_BY_GATES
    folder = str(run) if not isinstance(run, RunName) else str(name)
    try:
        import json
        record = json.loads((_DATAFILES / folder / "transpiled_circuits" / "physical_qubits.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return name.layout
    return name.layout if "decoherence_included" in record else LAYOUT_BY_GATES
_BENCH  = re.compile(r"[a-z0-9][a-z0-9-]*(\+[a-z]+-[0-9][0-9-]*)*")
_DATE   = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"#[1-9]\d*")


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
    date:      str | None = None   # YYYY-MM-DD, None for a name without a date
    run_number: int | None = None  # N of _#N, the run's number on that day; None with no date

    def __str__(self):
        return (f"{self.undated}_{self.date}_#{self.run_number}" if self.date else self.undated)

    @property
    def undated(self):
        """The name without its date: the run's settings, which dated_run_folder() looks up by."""
        return (f"{self.benchmark}_{self.data_type}_{self.provider}-{self.backend}_lnn-{self.lnn}"
                f"_service-{self.service}_spam-{self.spam}_layout-{self.layout}")

    @property
    def date_label(self):
        """The run's date and number as the figures show them, DD/MM/YYYY #N ("date unknown" without a date)."""
        if not self.date:
            return "date unknown"
        y, m, d = self.date.split("-")
        return f"{d}/{m}/{y} #{self.run_number}"

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
                "sim-dummy":     "dummy emulator (simulation)",
                "sim-fake-fez":  "simulation (FakeFez noise)",
                "sim-qpu-noise": "simulation (calibration noise)"}[self.data_type]
        return f"{name} — {kind}"

    @property
    def figure_label_full(self):
        """figure_label with the layout and the run's date, e.g. "ibm-fez — real QPU, layout optimized, 11/10/2026 #1"."""
        return f"{self.figure_label}, layout {layout_label(self)}, {self.date_label}"

    @property
    def is_dummy(self):
        """True for a dummy emulator of a QPU (sim-dummy): a coherent-noise simulation whose QPU time and cost are modelled."""
        return self.data_type == "sim-dummy"

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


def local_run_name(benchmark, data_type, lnn, do_spam=True, device="local-aer", layout_tag=None):
    """The run folder of a local-simulation script (no QPU: layout none, except sim-fake-fez, which requires
    layout_tag: LayoutArgs.tag, layout-optimized, … or the bare mode)."""
    if data_type == "sim-fake-fez" and layout_tag is None:
        raise ValueError("a sim-fake-fez run name needs a layout_tag (LayoutArgs.tag)")
    provider, b = device.split("-", 1)
    layout = "none" if layout_tag is None else layout_tag.removeprefix("layout-")
    return str(RunName(benchmark, data_type, provider, b, _flag(lnn), "local", _flag(do_spam), layout))


def parse_run_name(name):
    """RunName of a run-folder name (or a path ending in one); ValueError if it doesn't follow the scheme."""
    base   = str(name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    fields = base.split("_")
    date = run_number = None
    if len(fields) == 9:
        number = fields.pop()
        date   = fields.pop()
        if not _DATE.fullmatch(date):
            raise ValueError(f"{base!r}: bad date {date!r} (YYYY-MM-DD)")
        if not _NUMBER.fullmatch(number):
            raise ValueError(f"{base!r}: bad run number {number!r} (#1, #2, ...)")
        run_number = int(number[1:])
    if len(fields) != 7:
        raise ValueError(f"{base!r} is not a run name: expected 7 fields (9 with the date and run number) separated by '_', got {len(fields)}")
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
    return RunName(bench, dtype, provider, backend, lnn, service, spam, layout, date, run_number)


@dataclass(frozen=True)
class RunFilter:
    """Which runs run_post_process.py analyzes, beyond its selection: every field given must match (AND), and a
    list matches if any of its values does (OR). None = no restriction on that field.

    date_from / date_to   inclusive YYYY-MM-DD bounds (either may be None); a name without a date never matches
    backends              <provider>-<backend> (RunName.device), e.g. ibm-fez
    services              the service field, e.g. ibm-cloud
    layouts               the layout field, e.g. optimized, random, 15-14-21"""
    date_from: str | None = None
    date_to:   str | None = None
    backends:  tuple[str, ...] | None = None
    services:  tuple[str, ...] | None = None
    layouts:   tuple[str, ...] | None = None

    @property
    def is_empty(self):
        return all(v is None for v in (self.date_from, self.date_to, self.backends, self.services, self.layouts))

    def matches(self, name):
        """Whether the RunName `name` passes every filter that is set."""
        if self.date_from or self.date_to:
            if not name.date or (self.date_from and name.date < self.date_from) or (self.date_to and name.date > self.date_to):
                return False
        return ((self.backends is None or name.device in self.backends)
                and (self.services is None or name.service in self.services)
                and (self.layouts is None or name.layout in self.layouts or layout_label(name) in self.layouts))

    def describe(self):
        """The filter for the run list header, e.g. "date 2026-09-01 to 2026-09-30; backends ibm-fez, iqm-garnet"."""
        parts = []
        if self.date_from or self.date_to:
            parts.append("date " + _date_text(self, " to ", "from ", "until "))
        for label, values in (("backends", self.backends), ("services", self.services), ("layouts", self.layouts)):
            if values is not None:
                parts.append(f"{label} {', '.join(values)}")
        return "; ".join(parts) or "none"

    def suffix(self):
        """The part of a collective folder name that tells a filtered set from the unfiltered one, "" without
        filters, e.g. _dates-2026-09-01-to-2026-09-30_backends-ibm-fez+iqm-garnet. A long list becomes
        <count>-<hash> (Windows' 260-character path limit)."""
        parts = []
        if self.date_from or self.date_to:
            single = self.date_from == self.date_to
            parts.append(("date-" if single else "dates-") + _date_text(self, "-to-", "from-", "until-"))
        for label, values in (("backends", self.backends), ("services", self.services), ("layouts", self.layouts)):
            if values is not None:
                text = "+".join(sorted(values))
                if len(text) > 40:
                    text = f"{len(values)}-{hashlib.sha1(text.encode()).hexdigest()[:8]}"
                parts.append(f"{label}-{text}")
        return "".join("_" + p for p in parts)


def make_run_filter(date=None, backends=None, services=None, layouts=None):
    """A RunFilter from the command-line values: `date` as parse_date_filter() reads it, the others lists of
    strings (None = not given). ValueError naming the bad value if a date, backend, service or layout is not
    one a run name can have."""
    date_from, date_to = parse_date_filter(date) if date is not None else (None, None)
    for b in backends or ():
        if not re.fullmatch(r"[a-z0-9]+-[a-z0-9][a-z0-9-]*", b):
            raise ValueError(f"bad backend {b!r} (<provider>-<backend>, e.g. ibm-fez)")
    for s in services or ():
        if s not in SERVICES:
            raise ValueError(f"unknown service {s!r} (one of {', '.join(SERVICES)})")
    for l in layouts or ():
        if not _LAYOUT.fullmatch(l) and l not in LAYOUT_LABELS:
            raise ValueError(f"bad layout {l!r} (optimized, random, lowest, legacy, none, three qubits a-b-c, {', '.join(LAYOUT_LABELS)})")
    as_tuple = lambda v: tuple(v) if v else None
    return RunFilter(date_from, date_to, as_tuple(backends), as_tuple(services), as_tuple(layouts))


def _date_text(f, between, from_, until):
    if f.date_from and f.date_to:
        return f.date_from if f.date_from == f.date_to else f"{f.date_from}{between}{f.date_to}"
    return f"{from_}{f.date_from}" if f.date_from else f"{until}{f.date_to}"


def parse_date_filter(text):
    """(date_from, date_to) of a --date value: YYYY-MM-DD (that day), A:B (inclusive range), A: or :B (open end).
    ValueError if it is none of these or the range is reversed."""
    start, colon, end = text.partition(":")
    start, end = start or None, (end or None) if colon else start or None
    if (start is None and end is None) or any(d is not None and not _is_calendar_date(d) for d in (start, end)):
        raise ValueError(f"bad date {text!r} (YYYY-MM-DD, YYYY-MM-DD:YYYY-MM-DD, YYYY-MM-DD: or :YYYY-MM-DD)")
    if start and end and start > end:
        raise ValueError(f"bad date range {text!r}: the start is after the end")
    return start, end


def _is_calendar_date(text):
    from datetime import date
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return bool(_DATE.fullmatch(text))


def today_utc():
    """Today's UTC date, YYYY-MM-DD — the date field of a run folder made now."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _dated_key(name, folder_name):
    """(date, run number) of `folder_name` if it is <name>_<date>_#<N>, else None."""
    date, _, number = folder_name[len(name) + 1:].partition("_")
    if _DATE.fullmatch(date) and _NUMBER.fullmatch(number):
        return date, int(number[1:])
    return None


def dated_run_folders(undated, root="datafiles"):
    """The folders <root>/<undated>_<YYYY-MM-DD>_#<N> that exist, by date then run number (oldest first; the
    number sorts numerically). `undated` may carry a subfolder (tests/<name>); the returned names keep it."""
    undated = str(undated).replace("\\", "/")
    parent, _, name = undated.rpartition("/")
    folder = Path(root, parent) if parent else Path(root)
    if not folder.is_dir():
        return []
    keyed = [(key, p.name) for p in folder.glob(f"{name}_*")
             if p.is_dir() and (key := _dated_key(name, p.name))]
    dated = [n for _, n in sorted(keyed)]
    return [f"{parent}/{d}" if parent else d for d in dated]


def run_finished(run_folder, root="datafiles"):
    """True when a run folder needs nothing more from its submit / simulate / health-check script: every chunk
    of its run_info.json submitted (a real run) or collected (a simulation: every chunk has counts in
    raw_results/results_counts.json; a crash that left the last chunk's fidelity or SPAM counts incomplete still counts
    as finished, so a new run folder is started: use --force-new-run or delete the folder to redo it); a standalone
    health check once its job_records/health_check.json exists. A folder without run_info.json (nothing recorded yet) is unfinished."""
    import json
    path = Path(root, run_folder)
    name = parse_run_name(run_folder)
    if name.is_health_check:
        return (path / "job_records" / "health_check.json").exists()
    try:
        sample_size = json.loads((path / "run_info.json").read_text(encoding="utf-8"))["sample_size"]
    except (OSError, ValueError, KeyError):
        return False
    if name.is_simulation:
        from source.core.records.chunk_scope import collected_chunks
        return len(collected_chunks(path)) >= sample_size
    from source.core.records.job_records import chunks_without_submission
    return not chunks_without_submission(path, sample_size)


def run_has_uncollected(run_folder, root="datafiles"):
    """True when a run folder has something a collect script can still collect: a chunk job record in
    job_records/ whose status is neither "complete" nor "failed" (such as "submitted",
    or a record with no status), or a health check (the
    standalone job_records/health_check.json, or an accompanying health_checks/health_check_<N>.json) in that
    state. A failed job or check is not collectable; a folder without records is not collectable either."""
    from source.core.common import read_json_or_none
    from source.core.records.health_check_records import load_health_checks
    from source.core.records.job_records import chunk_record_files
    path = Path(root, run_folder)
    final = ("complete", "failed")
    files = list(chunk_record_files(path / "job_records").values()) + [path / "job_records" / "health_check.json"]
    for p in files:
        rec = read_json_or_none(p) if p.exists() else None
        if isinstance(rec, dict) and rec.get("status") not in final:
            return True
    return any(c.get("status") not in final for c in load_health_checks(path))


def dated_run_folder(undated, root="datafiles", new_run=False):
    """The run folder a submit / simulate / health-check script works in, for the name of its settings
    (qpu_run_name(), local_run_name(), health_check_run_name()): the latest <undated>_<date>_#<N> folder while it is
    unfinished (run_finished(); a resumed or partly failed run continues there, whatever its date), else
    <undated>_<today, UTC>_#<next number of today> (a new run). new_run=True (--force-new-run) always makes a new
    run. Creates nothing."""
    existing = dated_run_folders(undated, root)
    if existing and not new_run and not run_finished(existing[-1], root):
        return existing[-1]
    today = today_utc()
    name  = str(undated).replace("\\", "/").rpartition("/")[2]
    numbers = [k[1] for f in existing if (k := _dated_key(name, f.rpartition("/")[2])) and k[0] == today]
    return f"{undated}_{today}_#{max(numbers, default=0) + 1}"


def latest_run_folder(undated, root="datafiles", date=None):
    """The latest existing <undated>_<date>_#<N> folder, for the scripts that read a run (collect, job status);
    None if there is none. `date`: YYYY-MM-DD (the latest number of that day) or YYYY-MM-DD_#N (exactly that run)."""
    existing = dated_run_folders(undated, root)
    if date is not None:
        exact = "_#" in date
        existing = [f for f in existing if f.endswith("_" + date) or (not exact and f.rpartition("_#")[0].endswith("_" + date))]
    return existing[-1] if existing else None


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
    """summary_<run name>.json — the copy of a run's summary in its plots+summaries folder and in the collective ones.
    n_samples: a subset analysis of the run's first n samples (+nsamples-n in <benchmark>)."""
    return f"summary_{parse_run_name(run_folder).with_samples(n_samples)}.json"


PLOTS_FOLDER = "plots+summaries"   # analysis/plots+summaries/: every run's plots and summary json, and the collective sets
TESTS_FOLDER = "tests"   # datafiles/tests/ holds the test suite's runs and caches; analysis/tests/ its plots and summaries (plots+summaries/)


def is_test_run(run_folder):
    """True for a run folder of the test suite: tests/<run name> (relative to datafiles/)."""
    return str(run_folder).replace("\\", "/").split("/", 1)[0] == TESTS_FOLDER


def _analysis_root(run_folder):
    return Path("analysis", TESTS_FOLDER) if is_test_run(run_folder) else Path("analysis")


def plots_dir(run_folder):
    """A run's plots+summaries folder, relative to the project root: analysis/plots+summaries/<run folder>
    (old_runs_full/… keeps its subfolder), or analysis/tests/plots+summaries/<run name> for a test run."""
    rel = str(run_folder).replace("\\", "/")
    if is_test_run(rel):
        rel = rel.split("/", 1)[1]
    return _analysis_root(run_folder) / PLOTS_FOLDER / rel


def collective_plots_dir(name, test=False):
    """A collective plots+summaries folder (collective_folder_name()), under analysis/plots+summaries/ or, for test
    runs, analysis/tests/plots+summaries/. It holds the plots, the run table and the runs' summary jsons."""
    return Path("analysis", *([TESTS_FOLDER] if test else []), PLOTS_FOLDER, name)


PER_QPU_SUFFIXES = {"latest": "_latest-per-qpu", "best-nacc": "_best-nacc-per-qpu", "best-emin-at-nmax": "_best-emin-at-nmax-per-qpu"}


def collective_folder_name(benchmark, data_types, run_filter=None, latest_per_qpu=False, per_qpu=None):
    """<benchmark>_collective_<data type>[+<data type>...][<RunFilter.suffix()>][_latest-per-qpu | _best-nacc-per-qpu |
    _best-emin-at-nmax-per-qpu] for the plots comparing several runs. `per_qpu` is a key of PER_QPU_SUFFIXES
    (latest_per_qpu=True is per_qpu="latest")."""
    per_qpu = per_qpu or ("latest" if latest_per_qpu else None)
    return (f"{benchmark}_collective_{'+'.join(data_types)}{run_filter.suffix() if run_filter else ''}"
            f"{PER_QPU_SUFFIXES[per_qpu] if per_qpu else ''}")
