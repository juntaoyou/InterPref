"""Shared query-pool loader for the InterPref pipeline.

The experiments originally read the query pools from Hugging Face ``save_to_disk``
directories with ``load_from_disk(path)['q']``.  The submission package ships the same
four query pools as JSONL/CSV (see ``../dataset/``), so every pipeline script loads them
through :func:`load_queries`, which understands all of the historical formats:

* a ``save_to_disk`` directory with a ``q`` (or ``query``/``question``) column
* a ``.jsonl`` file whose objects carry ``query`` (or ``question``/``q``)
* a ``.csv`` file with a ``query`` (or ``question``/``q``) column

The returned list of strings is a drop-in replacement for ``load_from_disk(...)['q']``.
"""

import json
import os

try:  # optional - only needed for the save_to_disk format
    from datasets import load_from_disk as _hf_load_from_disk
except Exception:  # pragma: no cover
    _hf_load_from_disk = None


def _first_column(table, candidates):
    for c in candidates:
        if c in table.column_names:
            return table[c]
    raise KeyError(f"none of {candidates} found in columns {table.column_names}")


def load_queries(path):
    """Return the list of query strings stored at ``path`` (see module docstring)."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"query pool not found: {path}")

    if os.path.isdir(path):
        if _hf_load_from_disk is None:
            raise RuntimeError(
                "install the 'datasets' package to read save_to_disk query pools")
        table = _hf_load_from_disk(path)
        return list(_first_column(table, ["q", "query", "question"]))

    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json"):
        queries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in ("query", "question", "q"):
                    if key in obj:
                        queries.append(obj[key])
                        break
                else:
                    raise KeyError(f"no query field in JSONL line: {line[:80]}")
        return queries

    if ext == ".csv":
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for key in ("query", "question", "q"):
                if key in (reader.fieldnames or []):
                    return [row[key] for row in reader]
        raise KeyError(f"no query column in {path}; columns={reader.fieldnames}")

    raise ValueError(f"unsupported query-pool format: {path}")
