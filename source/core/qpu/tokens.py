"""
tokens.py
The provider API tokens, kept in <project root>/tokens/ (git-ignored, see
tokens/README.md). Every script reads its token through read_token() (or, for a
token kept as a file, like VLQ's LEXIS token, finds it through token_path()), so the
folder is defined in one place.
"""

from source.core.common import PROJECT_ROOT, exit_with_error

TOKENS_DIR = PROJECT_ROOT / "tokens"


def token_path(name):
    """Path of the credential file `name` (e.g. "ibm_token", "iqm_vlq_project_info") in tokens/."""
    return TOKENS_DIR / name


def read_token(name):
    """The token saved in tokens/<name>, stripped. Exits with a one-line explanation
    (code 2) when the file is missing, instead of a FileNotFoundError traceback."""
    path = token_path(name)
    if not path.exists():
        exit_with_error(f"No API token: tokens/{name} does not exist. Save the provider's token in that "
                        f"file — tokens/README.md says how it must look and where to get it.")
    return path.read_text(encoding="utf-8").strip()
