"""kill python processes whose argv contains the given script name (never the caller's shell)."""
import os, sys, signal
me = {os.getpid(), os.getppid()}
pat = sys.argv[1]
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) in me: continue
    try: cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except OSError: continue
    if cmd and cmd[0].endswith(b"python3") and any(pat.encode() in c for c in cmd[1:2]):
        os.kill(int(pid), signal.SIGTERM); print("killed", pid, b" ".join(cmd)[:80])
