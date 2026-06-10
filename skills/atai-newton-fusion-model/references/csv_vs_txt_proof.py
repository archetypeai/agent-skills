"""
Proof: on Newton C 2.6 /query, a file uploaded as text/csv does NOT reach the
model's context, but the IDENTICAL bytes uploaded as text/plain DO.

Method (controlled — only the file extension/content-type differs):
  * One header + one data row, whose prot column is "WIREGUARD" — a protocol
    the model would never produce by default (its priors are TCP/HTTP/DNS).
  * Write the content to local files (proof_input.txt / proof_input.csv in the
    current dir) so you can inspect/diff them, then upload each via the
    official files API (content-type is inferred from the extension).
  * Download each back and hash it, to prove the stored bytes are
    byte-identical — i.e. .csv is not mangled on upload; only the
    server-recorded content-type differs.
  * Ask "what is the value in the prot column?" N times each.
  * If the model answers WIREGUARD, it actually read the file. If it answers
    something else (TCP/HTTP/...), it's confabulating from priors — the file
    content never reached it.

Run:
    cd skills/atai-newton-fusion-model/references
    python csv_vs_txt_proof.py   # ATAI_API_KEY + ATAI_API_ENDPOINT from env/.env

Expected:
    .txt  -> WIREGUARD on (nearly) every trial
    .csv  -> never WIREGUARD (generic priors instead)
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from archetypeai.api_client import ArchetypeAI

from _common import make_client, query

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
    with open(path, "w") as file_handle:
        file_handle.write(content)
    return path


def download_stored_bytes(client: ArchetypeAI, file_id: str) -> bytes:
    """Download the stored file via the official files API and return its bytes."""
    with tempfile.TemporaryDirectory() as download_dir:
        local_copy = Path(download_dir) / file_id
        if not client.files.local.download(file_id, str(local_copy)):
            return b""
        return local_copy.read_bytes()


def server_file_type(client: ArchetypeAI, file_uid: str) -> str:
    """The content-type the server recorded for the upload."""
    metadata = client.requests_get(
        f"{client.api_endpoint}/files/metadata", params={"file_uid": file_uid}
    )
    if isinstance(metadata, list) and metadata:
        metadata = metadata[0]
    if isinstance(metadata, dict):
        return metadata.get("file_type", "?")
    return "?"


def ask(client: ArchetypeAI, file_id: str) -> str:
    text, _, _ = query(
        client,
        user_query=QUESTION,
        instruction_prompt="Answer only from the attached data.",
        file_ids=[file_id],
        max_new_tokens=20,
    )
    return text.strip()


def run(client: ArchetypeAI, suffix: str) -> None:
    local_path = write_local(CONTENT, suffix)
    upload_response = client.files.local.upload(local_path)
    file_id = upload_response["file_id"]
    file_uid = upload_response.get("file_uid", "")
    stored_bytes = download_stored_bytes(client, file_id)
    byte_identical = stored_bytes == CONTENT.encode()
    print(f"\n=== {suffix} (file_id={file_id}) — ground truth prot=WIREGUARD ===")
    print(f"  local file       : {local_path}")
    print(f"  server file_type : {server_file_type(client, file_uid)}")
    print(f"  stored bytes     : {len(stored_bytes)} | sha256={hashlib.sha256(stored_bytes).hexdigest()[:16]} "
          f"| byte-identical to local: {byte_identical}")
    print(f"  stored content   : {stored_bytes.decode(errors='replace')!r}")
    hits = 0
    for trial_index in range(TRIALS):
        answer = ask(client, file_id)
        read_file = "WIREGUARD" in answer.upper()
        hits += read_file
        print(f"  trial {trial_index + 1}: read_file={read_file}  answer={answer!r}")
    print(f"  >>> {hits}/{TRIALS} trials actually read the file content")


if __name__ == "__main__":
    shared_client = make_client()
    local_sha = hashlib.sha256(CONTENT.encode()).hexdigest()[:16]
    print("Identical content uploaded two ways; only the extension/content-type differs.")
    print(f"local content: {len(CONTENT)} bytes, sha256={local_sha}")
    run(shared_client, ".txt")
    run(shared_client, ".csv")
    print(
        "\nIf both stored files are byte-identical (same sha256) yet .txt reads WIREGUARD "
        "and .csv does not, the difference is purely content-type routing: the model is not "
        "receiving text/csv file content on /query. Upload tabular data as text/plain."
    )
