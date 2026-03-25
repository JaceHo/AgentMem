#!/usr/bin/env python3
"""Register OpenClaw skills and Claude Code agents to AgentMem capability index."""
import os, re, json, glob, sys, urllib.request, urllib.error

AGENTMEM = "http://127.0.0.1:18800"

def post(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{AGENTMEM}/register-tools",
        data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())

def parse_frontmatter(content, fallback_name):
    name = fallback_name
    description = f"OpenClaw skill: {fallback_name}"
    fm = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if fm:
        body = fm.group(1)
        nm = re.search(r'^name:\s*(.+)$', body, re.MULTILINE)
        ds = re.search(r'^description:\s*(.+)$', body, re.MULTILINE)
        if nm: name = nm.group(1).strip().strip("\"'")
        if ds: description = ds.group(1).strip().strip("\"'")
    else:
        h = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h: description = h.group(1).strip()
    return name, description

# --- OpenClaw skills ---
print("Registering OpenClaw skills...")
skills_dir = os.path.expanduser("~/.openclaw/skills")
skills = []
for entry in sorted(os.listdir(skills_dir)):
    skill_md = os.path.join(skills_dir, entry, "SKILL.md")
    if not os.path.isfile(skill_md):
        continue
    with open(skill_md, encoding="utf-8") as f:
        content = f.read()
    name, description = parse_frontmatter(content, entry)
    description = description  # may be long; truncate for display only
    print(f"  + {name}")
    skills.append({"name": name, "description": description,
                   "category": "skill", "source": "openclaw"})

result = post({"agent_id": "openclaw-skills", "replace_all": True, "tools": skills})
print(f"  queued: {result.get('tool_count')} skills\n")

# --- Claude Code agents ---
print("Registering Claude Code agents...")
agents_dir = os.path.expanduser("~/.claude/agents")
agents = []
for path in sorted(glob.glob(os.path.join(agents_dir, "*.md"))):
    fallback = os.path.basename(path).replace(".md", "")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    name, description = parse_frontmatter(content, fallback)
    description = description.replace("OpenClaw skill", "Claude Code agent")
    print(f"  + {name}")
    agents.append({"name": name, "description": description,
                   "category": "agent", "source": "claude-code"})

result = post({"agent_id": "claude-code-agents", "replace_all": True, "tools": agents})
print(f"  queued: {result.get('tool_count')} agents\n")

# --- Stats ---
with urllib.request.urlopen(f"{AGENTMEM}/stats", timeout=5) as r:
    d = json.loads(r.read())
print(f"AgentMem: tools={d['tools']}  procedures={d['procedures']}  episodes={d['episodes']}")
