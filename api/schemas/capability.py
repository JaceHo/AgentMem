"""API schemas for capability (tool/env/procedure) endpoints."""

from pydantic import AliasChoices, BaseModel, Field, field_validator


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
    user_id: str = ""                   # track the row (and its dedup lookups) belongs to
    agent_id: str = ""


class RecallProceduresRequest(RecallToolsRequest):
    user_id: str = ""
    agent_id: str = ""


class ToolFeedbackRequest(BaseModel):
    # Hooks / plugins historically used tool_name, name, or tool.
    tool_name: str = Field(
        default="",
        description="Slugified tool name (e.g. web_search_prime)",
        validation_alias=AliasChoices("tool_name", "name", "tool"),
    )
    success: bool = Field(
        default=False,
        description="Did the tool invocation succeed?",
        validation_alias=AliasChoices("success", "ok", "succeeded"),
    )
    session_id: str = Field(
        default="",
        validation_alias=AliasChoices("session_id", "sessionId", "session"),
    )

    @field_validator("tool_name")
    @classmethod
    def _strip_tool_name(cls, v: str) -> str:
        return (v or "").strip()


class ToolSequenceRequest(BaseModel):
    sequence: list[str]                 # ordered list of tool names used in session
    session_id: str = ""


class ProcedureFeedbackRequest(BaseModel):
    task_prefix: str                    # first ~80 chars of task to identify procedure
    success: bool
    session_id: str = ""
