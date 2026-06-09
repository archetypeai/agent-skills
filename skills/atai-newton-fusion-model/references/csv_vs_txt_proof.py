"""
Proof: on Newton C 2.6 /query, a file uploaded as text/csv does NOT reach the
model's context, but the IDENTICAL bytes uploaded as text/plain DO.

Method (controlled — only the file extension/content-type differs):
  * One header + one data row, whose prot column is "WIREGUARD" — a protocol
    the model would never produce by default (its priors are TCP/HTTP/DNS).
  * Write the content to local files (proof_input.txt / proof_input.csv in the
    current dir) so you can inspect/diff them, then upload each.
  * Download each back (GET /files/download/{file_id}) and hash it, to prove
    the stored bytes are byte-identical — i.e. .csv is not mangled on upload;
    only the server-recorded content-type differs.
  * Ask "what is the value in the prot column?" N times each.
  * If the model answers WIREGUARD, it actually read the file. If it answers
    something else (TCP/HTTP/...), it's confabulating from priors — the file
    content never reached it.

Run:
    cd skills/atai-newton-fusion-model/references
    python csv_vs_txt_proof.py        # uses ATAI_API_KEY from ../../../.env or env

Expected:
    .txt  -> WIREGUARD on (nearly) every trial
    .csv  -> never WIREGUARD (generic priors instead)
"""

from __future__ import annotations

import hashlib
import os
import sys

import requests

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass

API_KEY = os.environ.get("ATAI_API_KEY")
if not API_KEY:
    sys.exit("Set ATAI_API_KEY (export it or put it in a .env up the tree).")
ENDPOINT = os.environ.get("ATAI_API_ENDPOINT", "https://api.u1.archetypeai.app/v0.5").rstrip("/")
if not ENDPOINT.endswith("/v0.5"):
    ENDPOINT += "/v0.5"
MODEL = "Newton::c2_6_8b_fp8_260424d7a55d5e"
TRIALS = 5

# Identical content for both uploads. Ground truth: prot == WIREGUARD.
CONTENT = (
    "time_utc,mac_a,mac_b,prot,tran,port_a,port_b,bytes_a,bytes_b,pkts_a,pkts_b\n"
    "2019-10-19T15:55:11Z,ebd1a7fa8544,e323b826aa71,WIREGUARD,17,51820,68,123,456,7,8\n"
)
QUESTION = "What is the value in the prot column of the attached data? Answer with one word."


def write_local(content: str, suffix: str) -> str:
    """Write the content to a named local file in the current directory so it
    can be inspected/diffed before upload. Returns the absolute path."""
    path = os.path.abspath(f"proof_input{suffix}")
    with open(path, "w") as f:
        f.write(content)
    return path


def upload(path: str) -> dict:
    with open(path, "rb") as fh:
        r = requests.post(
            f"{ENDPOINT}/files",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": (os.path.basename(path), fh)},  # content-type inferred from extension
            timeout=120,
        )
    r.raise_for_status()
    return r.json()  # {is_valid, file_id, file_uid}


def download(file_id: str) -> bytes:
    """GET /v0.5/files/download/{file_id} — streams the raw stored bytes."""
    r = requests.get(
        f"{ENDPOINT}/files/download/{file_id}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=120,
    )
    r.raise_for_status()
    return r.content


def server_file_type(file_uid: str) -> str:
    """GET /v0.5/files/metadata — the content-type the server recorded."""
    r = requests.get(
        f"{ENDPOINT}/files/metadata",
        headers={"Authorization": f"Bearer {API_KEY}"},
        params={"file_uid": file_uid},
        timeout=60,
    )
    if not r.ok:
        return "?"
    meta = r.json()
    if isinstance(meta, list) and meta:
        meta = meta[0]
    return meta.get("file_type", "?")


def extract(payload: dict) -> str:
    resp = payload.get("response")
    if isinstance(resp, dict):
        inner = resp.get("response")
        if isinstance(inner, list) and inner:
            return inner[0] or ""
    if isinstance(resp, list) and resp:
        return resp[0] or ""
    return str(resp)


def ask(file_id: str) -> str:
    body = {
        "query": QUESTION,
        "instruction_prompt": "Answer only from the attached data.",
        "file_ids": [file_id],
        "model": MODEL,
        "max_new_tokens": 20,
        "sanitize": False,
    }
    r = requests.post(
        f"{ENDPOINT}/query",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    return extract(r.json()).strip()


def run(suffix: str) -> None:
    local_path = write_local(CONTENT, suffix)
    up = upload(local_path)
    fid, fuid = up["file_id"], up.get("file_uid", "")
    stored = download(fid)
    same = stored == CONTENT.encode()
    print(f"\n=== {suffix} (file_id={fid}) — ground truth prot=WIREGUARD ===")
    print(f"  local file       : {local_path}")
    print(f"  server file_type : {server_file_type(fuid)}")
    print(f"  stored bytes     : {len(stored)} | sha256={hashlib.sha256(stored).hexdigest()[:16]} "
          f"| byte-identical to local: {same}")
    print(f"  stored content   : {stored.decode(errors='replace')!r}")
    hits = 0
    for i in range(TRIALS):
        out = ask(fid)
        read = "WIREGUARD" in out.upper()
        hits += read
        print(f"  trial {i + 1}: read_file={read}  answer={out!r}")
    print(f"  >>> {hits}/{TRIALS} trials actually read the file content")


if __name__ == "__main__":
    local_sha = hashlib.sha256(CONTENT.encode()).hexdigest()[:16]
    print("Identical content uploaded two ways; only the extension/content-type differs.")
    print(f"local content: {len(CONTENT)} bytes, sha256={local_sha}")
    run(".txt")
    run(".csv")
    print(
        "\nIf both stored files are byte-identical (same sha256) yet .txt reads WIREGUARD "
        "and .csv does not, the difference is purely content-type routing: the model is not "
        "receiving text/csv file content on /query. Upload tabular data as text/plain."
    )
