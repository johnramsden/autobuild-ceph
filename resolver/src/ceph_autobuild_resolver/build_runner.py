"""Build stage execution inside an LXD container.

Each public method builds the appropriate ``Stage`` from ``build_steps`` and
delegates to ``_run_stage``, which iterates through the steps, accumulates
output, and writes a per-stage log file after every step.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from . import display
from .build_steps import (
    Stage,
    build_stage,
    install_build_requirements_stage,
    install_dependencies_stage,
    prepare_tarball_stage,
)
from .config import Config
from .lxd import ExecResult, LXDManager

log = logging.getLogger(__name__)

_LOG_TAIL_LINES = 500
_ERROR_CONTEXT_BEFORE = 50
_ERROR_CONTEXT_AFTER = 50
_ERROR_TRAILING_TAIL = 100
# Word-boundary match on common failure markers, case-insensitive. Broad on
# purpose — false positives degrade to "model sees a useful chunk anyway" (no
# worse than the flat tail). Also covers C++ / CMake / dpkg-specific patterns
# that don't match the word-boundary form (e.g. "undefined reference").
_ERROR_RE = re.compile(
    r"\b(error|failed|fatal)\b"
    r"|undefined reference"
    r"|cannot find"
    r"|CMake Error"
    r"|No such file"
    r"|ImportError",
    re.IGNORECASE,
)
# debuild runs with set -x, echoing all environment variables at the start.
# Some of those variable VALUES contain the word "error" (e.g.
# DH_OVERIDDEN_COMMAND=@echo 'error: ...'). Skip such assignment lines so
# the first-error scan finds real build failures, not env-dump noise.
_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*=")


def first_error_line(log_tail: str) -> str:
    """Return the first recognisable error/failed/fatal line from a build log.

    Skips env-variable assignment lines from debuild's ``set -x`` preamble
    (e.g. ``DH_OVERIDDEN_COMMAND=@echo 'error: ...'``) so we find real build
    failures rather than env-dump noise.  Falls back to the last non-empty
    line if no marker is found.
    """
    for line in log_tail.splitlines():
        if _ENV_VAR_RE.match(line):
            continue
        if _ERROR_RE.search(line):
            return line.strip()[:200]
    return log_tail.strip().splitlines()[-1][:200] if log_tail.strip() else ""


def _build_log_excerpt(full_output: str) -> str:
    """Extract a focused excerpt from build output.

    If any line matches an error/failed/fatal marker, return ±50 lines
    around the first hit followed by the last 100 lines of the log
    (separated by a `--- snip ---` marker so the model can tell the
    sections apart). Otherwise return the flat 500-line tail.
    """
    lines = full_output.splitlines()
    first_hit: int | None = None
    for i, line in enumerate(lines):
        # Skip env-variable assignment lines from the debuild set -x preamble.
        if _ENV_VAR_RE.match(line):
            continue
        if _ERROR_RE.search(line):
            first_hit = i
            break

    if first_hit is None:
        return "\n".join(lines[-_LOG_TAIL_LINES:])

    around_start = max(0, first_hit - _ERROR_CONTEXT_BEFORE)
    around_end = min(len(lines), first_hit + _ERROR_CONTEXT_AFTER + 1)
    around = lines[around_start:around_end]
    trailing_start = max(around_end, len(lines) - _ERROR_TRAILING_TAIL)
    trailing = lines[trailing_start:]

    parts = [
        f"--- first error context (lines {around_start + 1}-{around_end}) ---",
        "\n".join(around),
    ]
    if trailing and trailing_start > around_end:
        parts.extend(
            [
                f"--- snip ({trailing_start - around_end} lines) ---",
                f"--- trailing tail (lines {trailing_start + 1}-{len(lines)}) ---",
                "\n".join(trailing),
            ]
        )
    return "\n".join(parts)


@dataclass
class BuildOutcome:
    """Result of a build attempt with enough context to feed back to the model."""

    ok: bool
    stage: str
    returncode: int
    log_path: str  # path *inside the container* of the captured log
    log_tail: str  # last N lines, ready to splice into a prompt


class BuildRunner:
    def __init__(self, lxd: LXDManager, cfg: Config) -> None:
        self._lxd = lxd
        self._cfg = cfg

    @property
    def cfg(self) -> Config:
        return self._cfg

    @property
    def lxd(self) -> LXDManager:
        return self._lxd

    # ------------------------------------------------------------------
    # Build stages
    # ------------------------------------------------------------------

    def install_dependencies(self, container: str) -> ExecResult:
        display.stage_header("install_dependencies")
        return self._run_stage(container, install_dependencies_stage(self._cfg))

    def prepare_tarball(self, container: str) -> ExecResult:
        display.stage_header("prepare_tarball")
        return self._run_stage(container, prepare_tarball_stage(self._cfg))

    def install_build_requirements(self, container: str) -> ExecResult:
        display.stage_header("install_build_requirements")
        return self._run_stage(
            container, install_build_requirements_stage(self._cfg)
        )

    def build(self, container: str) -> BuildOutcome:
        display.stage_header("build")
        stage = build_stage(self._cfg)
        result = self._run_stage(container, stage)
        log_path = f"{self._cfg.container_log_dir}/{stage.name}.log"
        tail = _build_log_excerpt(result.stdout)
        outcome = BuildOutcome(
            ok=result.ok,
            stage=stage.name,
            returncode=result.returncode,
            log_path=log_path,
            log_tail=tail,
        )
        display.build_result(outcome.ok, outcome.stage, outcome.log_tail)
        return outcome

    # ------------------------------------------------------------------
    # Resolver-specific helpers
    # ------------------------------------------------------------------

    def apply_patch_to_tree(self, container: str, diff_text: str) -> ExecResult:
        """Apply a unified diff to the current working tree without resetting first.

        Unlike apply_diff, this preserves all uncommitted working-tree changes
        (patch files, rule edits, etc.) made earlier in the same session. Used
        by the model's apply_patch tool so mid-session state is not wiped.
        """
        self._lxd.put_text(container, "/tmp/resolver.diff", diff_text)
        return self._lxd.exec(
            container,
            ["git", "apply", "--whitespace=nowarn", "/tmp/resolver.diff"],
            cwd=self._cfg.container_workdir,
        )

    def apply_diff(self, container: str, diff_text: str) -> ExecResult:
        """Apply a unified diff to a clean HEAD baseline inside the container.

        Used by validation to apply the model's final diff to a pristine
        container copy — the tree MUST be reset first so the diff applies
        against a known baseline. For mid-session applies (model's apply_patch
        tool), use apply_patch_to_tree instead to preserve prior changes.
        """
        self._lxd.put_text(container, "/tmp/resolver.diff", diff_text)
        # Reset working tree AND index to a clean HEAD baseline.
        # git checkout resets the working tree; git reset resets the index
        # (needed if apply_diff is called multiple times in the same container).
        self._lxd.exec_shell(
            container,
            f"cd {self._cfg.container_workdir} && "
            "git checkout HEAD -- . >/dev/null 2>&1; "
            "git reset HEAD -- . >/dev/null 2>&1",
            check=False,
        )
        # Remove untracked files in debian/ that the diff adds as new files.
        # git apply aborts (silently, exit 0) when a "new file" already exists
        # in the working tree, leaving every subsequent hunk unapplied.  This
        # happens when the pristine snapshot predates a fresh git clone (e.g.
        # patch files copied in before the resolver started).
        self._lxd.exec_shell(
            container,
            f"cd {self._cfg.container_workdir} && "
            "git ls-files -o debian/ | xargs -r rm -f",
            check=False,
        )
        return self._lxd.exec(
            container,
            [
                "git",
                "apply",
                "--index",
                "--whitespace=nowarn",
                "/tmp/resolver.diff",
            ],
            cwd=self._cfg.container_workdir,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_stage(self, container: str, stage: Stage) -> ExecResult:
        log_path = f"{self._cfg.container_log_dir}/{stage.name}.log"
        buf = ""

        for step in stage.steps:
            display.step_start(step.argv)
            result = self._lxd.exec(container, step.argv, cwd=step.workdir)
            display.step_done(result.ok, result.returncode, result.stdout + result.stderr)
            buf += f"=== step: {step.argv} ===\n{result.stdout}"
            if result.stderr:
                buf += result.stderr + "\n"
            self._lxd.put_text(container, log_path, buf)
            if not result.ok and not step.allow_failure:
                return ExecResult(result.returncode, buf, result.stderr)

        return ExecResult(0, buf, "")
