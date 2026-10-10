# API tokens and access files

This folder holds the credentials the QPU scripts use: one plain-text file each. Everything here
except this README is git-ignored, so keep the files here and nowhere else. Scripts read them only
through `source/core/qpu/tokens.py` (`read_token()` / `token_path()`). A missing file stops the script
with a one-line message naming the file (exit code 2).

| File | Provider | What goes in it | Where to get it |
|---|---|---|---|
| `ibm_token` | IBM Quantum Platform | the API key, one line | [IBM Quantum Platform](https://quantum.cloud.ibm.com/): create an API key on your account's dashboard |
| `iqm_resonance_token` | IQM Resonance | the API token, one line | [IQM Resonance](https://resonance.iqm.tech/): generate an API token in your account settings |
| `iqm_eqe_token` | IQM EQE1 (MQSS, LRZ) | the MQSS token, one line | LRZ's MQSS (Munich Quantum Software Stack) access for your EuroHPC project |
| `aqt_piast-q_token` | AQT PIAST-Q (PCSS) | the PCSS API token, one line | PCSS (PSNC) quantum access for your EuroHPC project |
| `iqm_vlq_token` | IQM VLQ (QaaS, IT4I) | the LEXIS refresh token | **don't create it**: LEXIS access is provided by the EuroHPC project, and the code saves the LEXIS-generated token automatically: the first VLQ script opens the LEXIS (MyAccessID) login in the browser and writes it, and it is renewed on every use (`.../IQM_VLQ/readme_iqm_vlq.md` §2) |
| `iqm_vlq_project_info` | IQM VLQ (QaaS, IT4I) | two lines, see below | the LEXIS portal page of the EuroHPC project that holds the VLQ allocation (LEXIS access is provided by the EuroHPC project) |

`...` stands for `source/benchmark_scripts/real_qpu_scripts`.

## Token files

A token file holds the token alone, without quotes, a `KEY =` prefix, or anything else. Spaces and
a newline around it are ignored. To create one (here for IBM; use the file name from the table):

Windows (PowerShell):

```powershell
Set-Content -Path tokens\ibm_token -Value "<your IBM token>" -NoNewline -Encoding ascii
```

Linux/macOS:

```bash
printf '%s' '<your IBM token>' > tokens/ibm_token
chmod 600 tokens/ibm_token
```

## `iqm_vlq_project_info`

The project ID and the computational resource name, as `KEY = VALUE` lines.
This is needed for logging into the LEXIS system to get a VLQ access token.
Both keys are required:

```
PROJECT = <LEXIS project short name, e.g. EU-XX-YYY>
RESOURCE = <computational resource name, e.g., VLQ-EU>
```

To use another project or resource for one call, pass them to
`source.core.qpu.iqm_vlq.lexis_backend(project, resource)` instead.
