import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import questionary
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.ckpt_manager import CkptManager, PassSpec, PassState, RunManifest
from src.config import Config, get_default_config
from src.trainer import TrainerRunner

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_ROOT_DIR = Path(os.environ.get("MAHJONG_GPT_ROOT", "runs"))
DEFAULT_RUN_ID = os.environ.get("MAHJONG_GPT_RUN_ID", "model-00")


# -------------------------
# Common helpers
# -------------------------
def _store(root_dir: Path, run_id: str) -> CkptManager:
    """Create a checkpoint store."""
    return CkptManager(root_dir=root_dir, run_id=run_id)


def _require_manifest(store: CkptManager) -> RunManifest:
    """Load manifest or raise a friendly error."""
    if not store.exists():
        raise RuntimeError(
            f"Manifest not found: {store.manifest_path}\n"
            "Please initialize the run first."
        )
    return store.load_manifest()


def _find_pass_rec(manifest: RunManifest, pass_id: int):
    """Return PassRecord or None."""
    return next((p for p in manifest.passes if p.spec.pass_id == pass_id), None)


def _print_manifest(manifest: RunManifest) -> None:
    """Pretty-print manifest summary."""
    console.print(
        Panel.fit(
            f"[bold]Run[/bold]  run_id=[cyan]{manifest.run_id}[/cyan]  "
            f"active_pass_id=[magenta]{manifest.active_pass_id}[/magenta]  "
            f"passes=[green]{len(manifest.passes)}[/green]"
        )
    )

    t = Table(title="Passes", show_lines=True)
    t.add_column("pass_id", justify="right")
    t.add_column("name")
    t.add_column("status")
    t.add_column("curr_ep", justify="right")
    t.add_column("total_ep", justify="right")
    t.add_column("best_metric", justify="right")
    t.add_column("last_ckpt_dir")
    t.add_column("init_from")

    for p in sorted(manifest.passes, key=lambda x: x.spec.pass_id):
        init_from = "-"
        if p.spec.init_from_pass_id is not None:
            src_ep = (
                "latest"
                if p.spec.init_from_episode is None
                else str(p.spec.init_from_episode)
            )
            init_from = f"{p.spec.init_from_pass_id}:{src_ep} ({p.spec.init_mode})"

        t.add_row(
            str(p.spec.pass_id),
            p.spec.name or "-",
            p.state.status,
            str(p.state.curr_episode),
            str(p.spec.total_episodes),
            f"{p.state.best_metric:.4f}",
            p.state.last_checkpoint_dir or "-",
            init_from,
        )
    console.print(t)


def _edit_json_in_editor(initial_text: str, suffix: str = ".json") -> str:
    """Open a temp file in $EDITOR and return edited content."""
    editor = os.environ.get("EDITOR", "").strip() or "vi"
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=suffix, delete=False, encoding="utf-8"
    ) as tf:
        tf.write(initial_text)
        tf.flush()
        tmp_path = tf.name

    try:
        subprocess.run([editor, tmp_path], check=False)
        return Path(tmp_path).read_text(encoding="utf-8")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _rewrite_pass_spec(
    store: CkptManager, manifest: RunManifest, spec: PassSpec
) -> None:
    """Update spec in manifest and write pass_{id}/spec.json.

    Note: PassSpec is intended immutable once a pass starts; this tool is manual/unsafe by design.
    """
    rec = _find_pass_rec(manifest, spec.pass_id)
    if rec is None:
        raise RuntimeError(f"Pass {spec.pass_id} not found.")
    rec.spec = spec
    store.save_manifest(manifest)
    store._write_pass_spec(spec)  # type: ignore[attr-defined]


def _pick_pass_id(manifest: RunManifest, prompt: str) -> Optional[int]:
    """Interactive pass selector."""
    if not manifest.passes:
        return None
    choices = [
        questionary.Choice(
            title=f"{p.spec.pass_id} - {p.spec.name or '-'} ({p.state.status}) "
            f"ep {p.state.curr_episode}/{p.spec.total_episodes}",
            value=p.spec.pass_id,
        )
        for p in sorted(manifest.passes, key=lambda x: x.spec.pass_id)
    ]
    pid = questionary.select(prompt, choices=choices).ask()
    return None if pid is None else int(pid)


# -------------------------
# Interactive tasks
# -------------------------
def _task_status(store: CkptManager) -> None:
    """Show run manifest status."""
    manifest = _require_manifest(store)
    _print_manifest(manifest)


def _task_init_run(store: CkptManager) -> None:
    """Initialize a new run (manifest + pass-0)."""
    if store.exists():
        overwrite = questionary.confirm(
            f"Manifest already exists at {store.manifest_path}. Overwrite (DANGEROUS)?",
            default=False,
        ).ask()
        if not overwrite:
            console.print("[yellow]Cancelled.[/yellow]")
            return

    notes = questionary.text("Run notes:", default="First run").ask() or "First run"
    pass_name = questionary.text("First pass name:", default="pass-0").ask() or "pass-0"
    save_every = int(questionary.text("save_every:", default="10").ask() or "10")

    cfg = get_default_config()
    total_eps = questionary.text(
        "total_episodes for pass-0:",
        default=str(cfg.training.pass_n_episodes),
    ).ask()
    if total_eps is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return
    total_eps_i = int(total_eps)

    first = PassSpec(
        pass_id=0,
        name=pass_name,
        config=cfg,
        total_episodes=total_eps_i,
        save_every=save_every,
        eval_every=0,
        notes="First pass",
    )
    manifest = store.init_new(first, notes=notes)
    console.print("[green]Initialized run.[/green]")
    _print_manifest(manifest)


def _task_set_active(store: CkptManager) -> None:
    """Select which pass is active (TrainerRunner starts here)."""
    manifest = _require_manifest(store)
    pid = _pick_pass_id(manifest, "Select active pass:")
    if pid is None:
        console.print("[yellow]No passes found.[/yellow]")
        return

    if not questionary.confirm(f"Set active_pass_id to {pid}?", default=True).ask():
        console.print("[yellow]Cancelled.[/yellow]")
        return

    manifest.active_pass_id = pid
    store.save_manifest(manifest)
    console.print(f"[green]Active pass set to {pid}[/green]")


def _task_append_pass(store: CkptManager) -> None:
    """Create a new pass (optionally bootstrap from another pass)."""
    manifest = _require_manifest(store)
    new_id = (
        (max(p.spec.pass_id for p in manifest.passes) + 1) if manifest.passes else 0
    )

    cfg_src = questionary.select(
        "Config source:",
        choices=[
            questionary.Choice("Default (get_default_config)", value="default"),
            questionary.Choice("Copy from existing pass", value="copy"),
            questionary.Choice(
                "Edit JSON in $EDITOR (start from default)", value="edit"
            ),
        ],
    ).ask()
    if cfg_src is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    if cfg_src == "copy":
        src_pid = _pick_pass_id(manifest, "Copy config from which pass?")
        if src_pid is None:
            console.print("[yellow]Cancelled.[/yellow]")
            return
        src_rec = _find_pass_rec(manifest, src_pid)
        if src_rec is None:
            console.print("[red]Source pass not found.[/red]")
            return
        cfg = Config.from_dict(src_rec.spec.config.to_dict())
    else:
        cfg = get_default_config()

    if cfg_src == "edit":
        edited = _edit_json_in_editor(
            cfg.to_json(indent=2, ensure_ascii=False),
            suffix=f".pass{new_id}.config.json",
        )
        try:
            cfg = Config.from_json(edited)
        except Exception as e:
            console.print("[red]Config validation failed; not saved.[/red]")
            console.print(str(e))
            return

    name = (
        questionary.text("New pass name:", default=f"pass-{new_id}").ask()
        or f"pass-{new_id}"
    )
    notes = questionary.text("Pass notes:", default="").ask() or ""

    total_eps = int(
        questionary.text(
            "total_episodes:", default=str(cfg.training.pass_n_episodes)
        ).ask()
        or str(cfg.training.pass_n_episodes)
    )
    save_every = int(questionary.text("save_every:", default="10").ask() or "10")

    bootstrap = questionary.confirm(
        "Bootstrap from existing pass checkpoint?", default=False
    ).ask()
    init_from_pass_id: Optional[int] = None
    init_from_episode: Optional[int] = None
    init_mode: str = "weights_only"

    if bootstrap:
        init_from_pass_id = _pick_pass_id(manifest, "Select source pass:")
        if init_from_pass_id is None:
            console.print("[yellow]Cancelled.[/yellow]")
            return

        ep_txt = questionary.text("Source episode (blank = latest):", default="").ask()
        if ep_txt is None:
            console.print("[yellow]Cancelled.[/yellow]")
            return
        ep_txt = ep_txt.strip()
        init_from_episode = int(ep_txt) if ep_txt else None

        init_mode = (
            questionary.select(
                "init_mode:",
                choices=[
                    questionary.Choice("weights_only", value="weights_only"),
                    questionary.Choice(
                        "full (weights+optim+rng+replay+schedulers)", value="full"
                    ),
                ],
            ).ask()
            or "weights_only"
        )

    spec = PassSpec(
        pass_id=new_id,
        name=name,
        config=cfg,
        total_episodes=total_eps,
        save_every=save_every,
        eval_every=0,
        notes=notes,
        init_from_pass_id=init_from_pass_id,
        init_from_episode=init_from_episode,
        init_mode=init_mode,  # type: ignore[arg-type]
    )

    if not questionary.confirm(
        f"Append pass {new_id} and set it active?", default=True
    ).ask():
        console.print("[yellow]Cancelled.[/yellow]")
        return

    store.append_pass(manifest, spec)
    console.print(f"[green]Appended pass {new_id} and set active.[/green]")
    _print_manifest(store.load_manifest())


def _task_edit_config(store: CkptManager) -> None:
    """Edit a pass config in $EDITOR and validate with Pydantic."""
    manifest = _require_manifest(store)
    pid = _pick_pass_id(manifest, "Select pass to edit config:")
    if pid is None:
        console.print("[yellow]No passes found.[/yellow]")
        return

    rec = _find_pass_rec(manifest, pid)
    if rec is None:
        console.print("[red]Pass not found.[/red]")
        return

    if rec.state.status != "pending":
        proceed = questionary.confirm(
            f"Pass {pid} status is '{rec.state.status}'. Edit anyway (DANGEROUS)?",
            default=False,
        ).ask()
        if not proceed:
            console.print("[yellow]Cancelled.[/yellow]")
            return

    edited = _edit_json_in_editor(
        rec.spec.config.to_json(indent=2, ensure_ascii=False),
        suffix=f".pass{pid}.config.json",
    )
    try:
        cfg = Config.from_json(edited)
    except Exception as e:
        console.print("[red]Config validation failed; changes NOT saved.[/red]")
        console.print(str(e))
        return

    new_spec = PassSpec(**{**rec.spec.model_dump(mode="python"), "config": cfg})
    _rewrite_pass_spec(store, manifest, new_spec)
    console.print(f"[green]Saved config for pass {pid}.[/green]")


def _task_reset_pass_state(store: CkptManager) -> None:
    """Reset pass state to pending (does not delete checkpoints)."""
    manifest = _require_manifest(store)
    pid = _pick_pass_id(manifest, "Select pass to reset state:")
    if pid is None:
        console.print("[yellow]No passes found.[/yellow]")
        return

    rec = _find_pass_rec(manifest, pid)
    if rec is None:
        console.print("[red]Pass not found.[/red]")
        return

    hard = bool(
        questionary.confirm(
            "Hard reset best_metric/last_ckpt_dir?", default=False
        ).ask()
    )

    if not questionary.confirm(
        f"Reset state for pass {pid} (status->pending, curr_episode->0)?",
        default=False,
    ).ask():
        console.print("[yellow]Cancelled.[/yellow]")
        return

    st: PassState = rec.state
    st.status = "pending"
    st.curr_episode = 0
    if hard:
        st.best_metric = float("-inf")
        st.last_checkpoint_dir = None

    store.update_pass_state(manifest, pid, st)
    console.print(f"[green]Reset state for pass {pid}.[/green]")


def _task_run(store: CkptManager) -> None:
    """Run TrainerRunner on the active pass (auto-resume)."""
    manifest = _require_manifest(store)
    _print_manifest(manifest)

    if not questionary.confirm(
        f"Run active pass {manifest.active_pass_id} now?",
        default=True,
    ).ask():
        console.print("[yellow]Cancelled.[/yellow]")
        return

    console.print(Panel.fit("[cyan]Starting TrainerRunner.run(manifest)[/cyan]"))
    TrainerRunner(store).run(manifest)
    console.print("[green]TrainerRunner finished.[/green]")

def _task_ruff_check(store: CkptManager) -> None:
    """Run ruff check on all Python files."""
    fix = questionary.confirm("Apply --fix automatically?", default=False).ask()
    cmd = ["ruff", "check", "src/", "tests/"]
    if fix:
        cmd.append("--fix")

    console.print(f"[cyan]Running: {' '.join(cmd)}[/cyan]")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        console.print("[green]ruff check passed.[/green]")
    else:
        console.print(f"[yellow]ruff check exited with code {result.returncode}[/yellow]")


def _task_ruff_format(store: CkptManager) -> None:
    """Run ruff format on all Python files."""
    cmd = ["ruff", "format", "src/", "tests/"]

    console.print(f"[cyan]Running: {' '.join(cmd)}[/cyan]")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        console.print("[green]ruff format completed.[/green]")
    else:
        console.print(f"[yellow]ruff format exited with code {result.returncode}[/yellow]")


def _task_pytest_cov(store: CkptManager) -> None:
    """Run pytest with coverage on src."""
    cmd = ["pytest", "--cov=src"]

    extra = questionary.text("Extra pytest args (optional):", default="").ask()
    if extra and extra.strip():
        cmd.extend(extra.strip().split())

    console.print(f"[cyan]Running: {' '.join(cmd)}[/cyan]")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        console.print("[green]pytest passed.[/green]")
    else:
        console.print(f"[red]pytest exited with code {result.returncode}[/red]")

TASKS: List[Dict[str, Any]] = [
    {
        "name": "status",
        "desc": "Show manifest status (passes, active pass, progress).",
        "fn": _task_status,
    },
    {
        "name": "init-run",
        "desc": "Initialize a new run (manifest + pass-0).",
        "fn": _task_init_run,
    },
    {
        "name": "set-active",
        "desc": "Select which pass is active (TrainerRunner starts here).",
        "fn": _task_set_active,
    },
    {
        "name": "append-pass",
        "desc": "Create a new pass (optionally bootstrap from another pass).",
        "fn": _task_append_pass,
    },
    {
        "name": "edit-config",
        "desc": "Edit a pass config in $EDITOR and validate with Pydantic.",
        "fn": _task_edit_config,
    },
    {
        "name": "reset-pass-state",
        "desc": "Reset pass state to pending (does not delete checkpoints).",
        "fn": _task_reset_pass_state,
    },
    {
        "name": "run",
        "desc": "Run TrainerRunner on the active pass (auto-resume).",
        "fn": _task_run,
    },
    {
        "name": "ruff-check",
        "desc": "Run ruff check **/*.py (optional --fix).",
        "fn": _task_ruff_check,
    },
    {
        "name": "ruff-format",
        "desc": "Run ruff format **/*.py.",
        "fn": _task_ruff_format,
    },
    {
        "name": "pytest-cov",
        "desc": "Run pytest --cov=src.",
        "fn": _task_pytest_cov,
    },
    {"name": "exit", "desc": "Exit the CLI tool.", "fn": None},
]


@app.command()
def interactive() -> None:
    """Start the interactive CLI."""
    console.print("[bold green]Mahjong-GPT CLI[/bold green]")

    root_txt = questionary.text("root_dir:", default=str(DEFAULT_ROOT_DIR)).ask()
    run_txt = questionary.text("run_id:", default=str(DEFAULT_RUN_ID)).ask()
    if root_txt is None or run_txt is None:
        return

    store = _store(Path(root_txt), str(run_txt))
    console.print(
        Panel.fit(
            f"root=[cyan]{store.root_dir}[/cyan]\n"
            f"run_id=[magenta]{store.run_id}[/magenta]\n"
            f"manifest=[dim]{store.manifest_path}[/dim]"
        )
    )

    while True:
        options = [
            questionary.Choice(title=f"{t['name']} - {t['desc']}", value=t["name"])
            for t in TASKS
        ]
        selected = questionary.select("Choose a task:", choices=options).ask()
        if selected is None:
            continue
        if selected == "exit":
            console.print("[bold blue]Bye.[/bold blue]")
            break

        task = next(x for x in TASKS if x["name"] == selected)
        confirm = questionary.confirm(f"Proceed with `{selected}`?", default=True).ask()
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            continue

        fn: Callable[[CkptManager], None] = task["fn"]
        try:
            fn(store)
        except Exception as e:
            console.print(Panel.fit("[red]Task failed[/red]"))
            console.print(str(e))


if __name__ == "__main__":
    app()
