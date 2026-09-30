"""`--background` for long-running scripts: relaunch under nohup (and caffeinate on macOS), then return.

The relaunched run survives closing the terminal (nohup), and on macOS keeps the Mac from
idle-sleeping until it ends (caffeinate -i). Output goes to a log file. Stdlib only.

    ap = argparse.ArgumentParser(...)
    add_background_flag(ap, default_log="out/optimize.log")
    args = ap.parse_args()
    maybe_detach(args)          # returns only in the foreground run
"""
import os
import shutil
import subprocess
import sys



def add_background_flag(ap, default_log):
    ap.add_argument("--background", nargs="?", const=default_log, metavar="LOG",
                    help=f"run detached under nohup (+ caffeinate on macOS), output to LOG (default {default_log})")


def _strip_flag(argv):
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == "--background":
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            skip = nxt is not None and not nxt.startswith("-")   # --background LOG
            continue
        if a.startswith("--background="):
            continue
        out.append(a)
    return out


def maybe_detach(args):
    """If --background was given, start the same command detached and exit this process."""
    if not getattr(args, "background", None):
        return
    cwd = os.getcwd()          # relative --roles / --out keep meaning the same folders
    log = os.path.abspath(args.background)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    keep_awake = ["caffeinate", "-i"] if shutil.which("caffeinate") else []
    cmd = ["nohup", *keep_awake, sys.executable, "-u", os.path.abspath(sys.argv[0]), *_strip_flag(sys.argv[1:])]
    with open(log, "ab") as fh:
        fh.write(f"\n=== {' '.join(cmd)}\n".encode())
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=cwd, start_new_session=True)
    rel = os.path.relpath(log, cwd) if log.startswith(cwd + os.sep) else log
    print(f"running in the background (pid {proc.pid}{', caffeinate keeps the Mac awake' if keep_awake else ''})")
    print(f"  follow:  tail -f {rel}")
    print(f"  stop:    kill {proc.pid}")
    sys.exit(0)
