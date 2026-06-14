"""API schemas for capability (tool/env/procedure) endpoints."""

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    description: str
    category: str = ""
    source: str = "builtin"             # builtin | mcp | plugin | skill
    parameters: list[str] = []


class RegisterToolsRequest(BaseModel):
    tools: list[ToolDefinition] = Field(max_length=100)
    agent_id: str = ""                  # optional agent identifier
    replace_all: bool = False           # if True, clear existing tools first


class EnvState(BaseModel):
    os: str = ""
    os_version: str = ""
    shell: str = ""
    cwd: str = ""
    git_repo: str = ""
    git_branch: str = ""
    runtime: str = ""                   # e.g. "python3.12", "node20"
    active_mcps: list[str] = []
    active_plugins: list[str] = []
    active_skills: list[str] = []
    agent_model: str = ""               # e.g. "claude-sonnet-4-6"
    agent_version: str = ""
    session_id: str = ""
    extra: dict = {}                    # any extra key/value pairs


class RecallToolsRequest(BaseModel):
    query: str
    k: int = 5
    category: str = ""                  # optional filter by tool category
    source: str = ""                    # optional filter by source


class StoreProcedureRequest(BaseModel):
    task: str                           # what kind of task (used as embedding key)
    procedure: str                      # the step-by-step procedure
    tools_used: list[str] = []          # tools/skills involved
    domain: str = ""
    session_id: str = ""


class ToolFeedbackRequest(BaseModel):
    tool_name: str                      # slugified tool name (e.g. "web_search_prime")
    success: bool                       # did the tool invocation succeed?
    session_id: str = ""


class ToolSequenceRequest(BaseModel):
    sequence: list[str]                 # ordered list of tool names used in session
    session_id: str = ""


class ProcedureFeedbackRequest(BaseModel):
    task_prefix: str                    # first ~80 chars of task to identify procedure
    success: bool
    session_id: str = ""
