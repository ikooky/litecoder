"""Public interfaces for the tasks package."""

from litecoder.tasks.graph import MissingDependency, TaskGraph
from litecoder.tasks.manager import (
    InvalidTaskTransition,
    TaskAlreadyExists,
    TaskBlocked,
    TaskManager,
    TaskManagerError,
    TaskNotClaimable,
    TaskNotFound,
    TaskOwnershipError,
)
from litecoder.tasks.models import TaskCreate, TaskRecord, TaskStatus
from litecoder.tasks.planning import (
    MissingTaskDependency,
    PlanningView,
    TaskCycleError,
)
from litecoder.tasks.store import TaskStore
from litecoder.tasks.worktrees import (
    ProjectGitLock,
    WorktreeBinding,
    WorktreeError,
    WorktreeManager,
)
from litecoder.tasks.subagents import (
    AgentCaller,
    AgentCreationDenied,
    ChildAgentHandle,
    ChildAgentRequest,
    ChildAuthority,
    SubagentManager,
)

__all__ = [
    "AgentCaller",
    "AgentCreationDenied",
    "ChildAgentHandle",
    "ChildAgentRequest",
    "ChildAuthority",
    "InvalidTaskTransition",
    "MissingDependency",
    "MissingTaskDependency",
    "PendingRequest",
    "PlanningView",
    "ProtocolManager",
    "ProtocolNotificationError",
    "ProtocolResponse",
    "ProtocolViolation",
    "TaskAlreadyExists",
    "TaskBlocked",
    "TaskCreate",
    "TaskCycleError",
    "TaskManager",
    "TaskManagerError",
    "TaskNotClaimable",
    "TaskNotFound",
    "TaskOwnershipError",
    "TaskRecord",
    "TaskStatus",
    "SubagentManager",
    "TaskGraph",
    "MessageBus",
    "TeamMessage",
    "TeamManager",
    "TeamMember",
    "TeamRoster",
    "TaskStore",
    "ProjectGitLock",
    "WorktreeBinding",
    "WorktreeError",
    "WorktreeManager",
]

from litecoder.tasks.message_bus import MessageBus, TeamMessage
from litecoder.tasks.protocols import (
    PendingRequest,
    ProtocolManager,
    ProtocolNotificationError,
    ProtocolResponse,
    ProtocolViolation,
)
from litecoder.tasks.teams import TeamManager, TeamMember, TeamRoster
