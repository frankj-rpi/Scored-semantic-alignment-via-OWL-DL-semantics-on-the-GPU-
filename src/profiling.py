from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Iterator, List, Optional
import csv
import json

import torch


@dataclass
class ProfileNode:
    name: str
    label: str
    elapsed_ms_inclusive: float = 0.0
    children: List["ProfileNode"] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def self_ms(self) -> float:
        return self.elapsed_ms_inclusive - sum(child.elapsed_ms_inclusive for child in self.children)

    def add_child(self, child: "ProfileNode") -> "ProfileNode":
        self.children.append(child)
        return child


@dataclass
class ProfileValidationWarning:
    path: str
    issue: str
    elapsed_ms: float
    tolerance_ms: float


@dataclass
class ProfileSummary:
    root_elapsed_ms: float
    category_totals_ms: Dict[str, float]
    warnings: List[ProfileValidationWarning] = field(default_factory=list)
    reconciliation_residual_ms: float = 0.0


class ProfileRecorder:
    def __init__(self) -> None:
        self._root: Optional[ProfileNode] = None
        self._stack: List[tuple[ProfileNode, float, bool]] = []

    def start_root(self, name: str, label: Optional[str] = None, **meta: object) -> ProfileNode:
        node = ProfileNode(name=name, label=label or name, meta=dict(meta))
        self._root = node
        self._stack = [(node, perf_counter(), False)]
        return node

    def push(
        self,
        name: str,
        label: Optional[str] = None,
        *,
        sync_cuda: bool = False,
        **meta: object,
    ) -> ProfileNode:
        if not self._stack:
            raise RuntimeError("ProfileRecorder.push() requires an active root.")
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        parent = self._stack[-1][0]
        node = parent.add_child(ProfileNode(name=name, label=label or name, meta=dict(meta)))
        self._stack.append((node, perf_counter(), sync_cuda))
        return node

    def pop(self) -> ProfileNode:
        if not self._stack:
            raise RuntimeError("ProfileRecorder.pop() called with empty stack.")
        node, started_at, sync_cuda = self._stack.pop()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        node.elapsed_ms_inclusive += (perf_counter() - started_at) * 1000.0
        return node

    @contextmanager
    def scoped(
        self,
        name: str,
        label: Optional[str] = None,
        *,
        sync_cuda: bool = False,
        **meta: object,
    ) -> Iterator[ProfileNode]:
        node = self.push(name, label, sync_cuda=sync_cuda, **meta)
        try:
            yield node
        finally:
            self.pop()

    def build_tree(self) -> Optional[ProfileNode]:
        while self._stack:
            self.pop()
        return self._root

    def snapshot(self) -> Optional[ProfileNode]:
        return self._root


def tolerance_ms(total_elapsed_ms: float) -> float:
    return max(50.0, total_elapsed_ms * 0.01)


def aggregate_by_category(root: ProfileNode) -> Dict[str, float]:
    totals: Dict[str, float] = {}

    def visit(node: ProfileNode) -> None:
        category = node.meta.get("category")
        if isinstance(category, str):
            totals[category] = totals.get(category, 0.0) + max(node.self_ms, 0.0)
        for child in node.children:
            visit(child)

    visit(root)
    return totals


def validate_profile_tree(root: ProfileNode) -> List[ProfileValidationWarning]:
    warnings: List[ProfileValidationWarning] = []

    def visit(node: ProfileNode, path: str) -> None:
        child_total = sum(child.elapsed_ms_inclusive for child in node.children)
        tol = tolerance_ms(node.elapsed_ms_inclusive)
        if child_total - node.elapsed_ms_inclusive > tol:
            warnings.append(
                ProfileValidationWarning(
                    path=path,
                    issue="children exceed parent inclusive time",
                    elapsed_ms=child_total - node.elapsed_ms_inclusive,
                    tolerance_ms=tol,
                )
            )
        for child in node.children:
            child_path = f"{path}/{child.name}" if path else child.name
            visit(child, child_path)

    visit(root, root.name)
    return warnings


def profile_tree_to_dict(root: ProfileNode, root_elapsed_ms: Optional[float] = None) -> Dict[str, object]:
    root_total = root_elapsed_ms if root_elapsed_ms is not None else root.elapsed_ms_inclusive

    def encode(node: ProfileNode, parent_elapsed_ms: Optional[float]) -> Dict[str, object]:
        parent_pct = (
            (node.elapsed_ms_inclusive / parent_elapsed_ms) * 100.0
            if parent_elapsed_ms and parent_elapsed_ms > 0.0
            else 100.0
        )
        root_pct = (node.elapsed_ms_inclusive / root_total) * 100.0 if root_total > 0.0 else 100.0
        return {
            "name": node.name,
            "label": node.label,
            "inclusive_ms": node.elapsed_ms_inclusive,
            "self_ms": node.self_ms,
            "percent_of_parent": parent_pct,
            "percent_of_root": root_pct,
            "meta": dict(node.meta),
            "children": [encode(child, node.elapsed_ms_inclusive) for child in node.children],
        }

    return encode(root, None)


def flatten_profile_tree(root: ProfileNode) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    root_total = root.elapsed_ms_inclusive

    def visit(node: ProfileNode, parent_path: str, depth: int, parent_elapsed_ms: Optional[float]) -> None:
        path = f"{parent_path}/{node.name}" if parent_path else node.name
        rows.append(
            {
                "path": path,
                "name": node.name,
                "label": node.label,
                "parent_path": parent_path,
                "depth": depth,
                "inclusive_ms": node.elapsed_ms_inclusive,
                "self_ms": node.self_ms,
                "percent_of_parent": (
                    (node.elapsed_ms_inclusive / parent_elapsed_ms) * 100.0
                    if parent_elapsed_ms and parent_elapsed_ms > 0.0
                    else 100.0
                ),
                "percent_of_root": (
                    (node.elapsed_ms_inclusive / root_total) * 100.0 if root_total > 0.0 else 100.0
                ),
                "category": node.meta.get("category", ""),
                "iteration": node.meta.get("iteration", ""),
                "device": node.meta.get("device", ""),
                "notes": node.meta.get("notes", ""),
            }
        )
        for child in node.children:
            visit(child, path, depth + 1, node.elapsed_ms_inclusive)

    visit(root, "", 0, None)
    return rows


def write_profile_json(path: str | Path, root: ProfileNode, summary: ProfileSummary) -> None:
    payload = {
        "tree": profile_tree_to_dict(root),
        "summary": {
            "root_elapsed_ms": summary.root_elapsed_ms,
            "category_totals_ms": summary.category_totals_ms,
            "reconciliation_residual_ms": summary.reconciliation_residual_ms,
            "warnings": [
                {
                    "path": warning.path,
                    "issue": warning.issue,
                    "elapsed_ms": warning.elapsed_ms,
                    "tolerance_ms": warning.tolerance_ms,
                }
                for warning in summary.warnings
            ],
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_profile_csv(path: str | Path, root: ProfileNode) -> None:
    rows = flatten_profile_tree(root)
    fieldnames = [
        "path",
        "name",
        "label",
        "parent_path",
        "depth",
        "inclusive_ms",
        "self_ms",
        "percent_of_parent",
        "percent_of_root",
        "category",
        "iteration",
        "device",
        "notes",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_profile_tree(
    root: ProfileNode,
    *,
    verbose: bool = False,
    include_category_rollup: bool = True,
    warnings: Optional[Iterable[ProfileValidationWarning]] = None,
) -> str:
    lines: List[str] = []
    root_total = root.elapsed_ms_inclusive
    cutoff_ms = max(1.0, root_total * 0.001)

    def render(node: ProfileNode, depth: int, parent_elapsed_ms: Optional[float]) -> None:
        if depth > 0 and not verbose and node.elapsed_ms_inclusive <= cutoff_ms and node.self_ms <= cutoff_ms:
            return
        indent = "  " * depth
        parent_pct = (
            (node.elapsed_ms_inclusive / parent_elapsed_ms) * 100.0
            if parent_elapsed_ms and parent_elapsed_ms > 0.0
            else 100.0
        )
        root_pct = (node.elapsed_ms_inclusive / root_total) * 100.0 if root_total > 0.0 else 100.0
        line = (
            f"{indent}- {node.label}: {node.elapsed_ms_inclusive / 1000.0:.3f} s "
            f"({parent_pct:.1f}% parent, {root_pct:.1f}% root)"
        )
        if node.self_ms > tolerance_ms(node.elapsed_ms_inclusive):
            line += f", self={node.self_ms / 1000.0:.3f} s"
        lines.append(line)
        for child in node.children:
            render(child, depth + 1, node.elapsed_ms_inclusive)

    render(root, 0, None)

    if include_category_rollup:
        category_totals = aggregate_by_category(root)
        if category_totals:
            lines.append("By category:")
            for category, elapsed_ms in sorted(category_totals.items(), key=lambda item: item[0]):
                lines.append(f"  - {category}: {elapsed_ms / 1000.0:.3f} s")

    warning_list = list(warnings or ())
    if warning_list:
        lines.append("Timing warnings:")
        for warning in warning_list:
            lines.append(
                "  - "
                f"{warning.path}: {warning.issue}; residual={warning.elapsed_ms / 1000.0:.3f} s, "
                f"tolerance={warning.tolerance_ms / 1000.0:.3f} s"
            )
    return "\n".join(lines)
