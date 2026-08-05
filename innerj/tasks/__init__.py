"""Task generators. Each family produces matched A/B/C/D conditions per instance."""

from innerj.tasks.base import (
    CONTRAST,
    Condition,
    Record,
    check_label_absent,
    complete_instances,
    group_by_instance,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    "CONTRAST",
    "Condition",
    "Record",
    "check_label_absent",
    "complete_instances",
    "group_by_instance",
    "read_jsonl",
    "write_jsonl",
]
