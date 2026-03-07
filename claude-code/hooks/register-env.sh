#!/usr/bin/env bash
# Hook: SessionStart — registers OS/git/cwd environment state with AgentMem.
unset PYTHONWARNINGS  # prevent invalid startup-time warning filter from spamming stderr
python3 - << 'PYEOF'
import json, subprocess, os, platform, urllib.request

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except:
        return ""

env = {
    "os": platform.system(),
    "os_version": platform.mac_ver()[0] or platform.version(),
    "shell": os.environ.get("SHELL", ""),
    "cwd": os.getcwd(),
    "git_repo": sh("git rev-parse --show-toplevel"),
    "git_branch": sh("git rev-parse --abbrev-ref HEAD"),
    "runtime": f"python{platform.python_version()}",
    "agent_model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
}

payload = json.dumps(env).encode()
req = urllib.request.Request(
    "http://localhost:18800/register-env",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    urllib.request.urlopen(req, timeout=3)
except:
    pass
PYEOF

exit 0
