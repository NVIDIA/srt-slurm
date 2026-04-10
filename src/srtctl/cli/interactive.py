#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interactive CLI for srtctl using Rich and Questionary.

This provides a user-friendly interactive mode for:
- Browsing and selecting recipe configs
- Previewing sweep expansions
- Modifying parameters before submission
- Syntax-highlighted sbatch script preview

Usage:
    srtctl                          # Launch interactive mode
    srtctl -i                       # Explicit interactive flag
    srtctl apply -f config.yaml     # Non-interactive (unchanged)
"""

import contextlib
from pathlib import Path
from typing import Any

import questionary
import yaml
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

console = Console()

# Custom questionary style
STYLE = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
    ]
)


def find_recipes(root: Path | None = None) -> list[Path]:
    """Find all recipe YAML files in the recipes directory."""
    if root is None:
        root = Path.cwd()

    recipes_dir = root / "recipes"  # Note: typo in original dir name
    if not recipes_dir.exists():
        recipes_dir = root / "recipes"

    if not recipes_dir.exists():
        return []

    return sorted(recipes_dir.rglob("*.yaml"))


def display_config_summary(config: dict[str, Any], title: str = "Configuration") -> None:
    """Display a rich summary of a config dict."""
    tree = Tree(f"[bold cyan]{title}[/]")

    # Model info
    if "model" in config:
        model_branch = tree.add("[bold]📦 Model[/]")
        m = config["model"]
        model_branch.add(f"path: [green]{m.get('path', 'N/A')}[/]")
        model_branch.add(f"container: [dim]{m.get('container', 'N/A')}[/]")
        model_branch.add(f"precision: [yellow]{m.get('precision', 'N/A')}[/]")

    # Resources
    if "resources" in config:
        res_branch = tree.add("[bold]🖥️  Resources[/]")
        r = config["resources"]
        res_branch.add(f"gpu_type: [cyan]{r.get('gpu_type', 'N/A')}[/]")
        res_branch.add(f"prefill: [green]{r.get('prefill_workers', r.get('prefill_nodes', 'N/A'))}[/] workers")
        res_branch.add(f"decode: [green]{r.get('decode_workers', r.get('decode_nodes', 'N/A'))}[/] workers")
        res_branch.add(f"gpus_per_node: [yellow]{r.get('gpus_per_node', 'N/A')}[/]")

    # Benchmark
    if "benchmark" in config:
        bench_branch = tree.add("[bold]📊 Benchmark[/]")
        b = config["benchmark"]
        bench_branch.add(f"type: [magenta]{b.get('type', 'N/A')}[/]")
        if "isl" in b:
            bench_branch.add(f"isl: {b['isl']}, osl: {b.get('osl', 'N/A')}")
        if "concurrencies" in b:
            bench_branch.add(f"concurrencies: {b['concurrencies']}")

    # Sweep params
    if "sweep" in config:
        sweep_branch = tree.add("[bold]🔄 Sweep Parameters[/]")
        for key, values in config["sweep"].items():
            sweep_branch.add(f"{key}: [cyan]{values}[/]")

    console.print(tree)


def display_sweep_table(configs: list[tuple[dict, dict]], title: str = "Sweep Jobs") -> None:
    """Display sweep configurations as a rich table."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Job Name", style="green")
    table.add_column("Parameters", style="yellow")

    for i, (config_dict, params) in enumerate(configs, 1):
        job_name = config_dict.get("name", f"job_{i}")
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        table.add_row(str(i), job_name, params_str)

    console.print(table)


def display_sbatch_script(script: str, title: str = "Generated sbatch Script") -> None:
    """Display sbatch script with syntax highlighting."""
    syntax = Syntax(script, "bash", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="cyan"))


def select_recipe() -> Path | None:
    """Interactive recipe selection."""
    recipes = find_recipes()

    choices = []

    if recipes:
        # Group by subdirectory
        by_dir: dict[str, list[Path]] = {}
        for r in recipes:
            rel = r.relative_to(Path.cwd())
            parent = str(rel.parent)
            if parent not in by_dir:
                by_dir[parent] = []
            by_dir[parent].append(r)

        for dir_name in sorted(by_dir.keys()):
            choices.append(questionary.Separator(f"── {dir_name} ──"))
            for r in by_dir[dir_name]:
                rel = r.relative_to(Path.cwd())
                choices.append(questionary.Choice(str(rel.name), value=r))

    if not choices:
        console.print("[yellow]No recipes found in recipes/[/]")
        custom = questionary.path(
            "Enter path to config file:",
            style=STYLE,
        ).ask()
        return Path(custom) if custom else None

    choices.append(questionary.Separator("──────────────"))
    choices.append(questionary.Choice("📁 Browse for file...", value="browse"))

    selection = questionary.select(
        "Select a recipe:",
        choices=choices,
        style=STYLE,
    ).ask()

    if selection == "browse":
        custom = questionary.path(
            "Enter path to config file:",
            style=STYLE,
        ).ask()
        return Path(custom) if custom else None

    return selection


def confirm_submission(config_path: Path, config: dict, is_sweep: bool) -> bool:
    """Show summary and confirm before submission."""
    console.print()
    console.print(
        Panel(
            f"[bold]{config.get('name', config_path.name)}[/]",
            title="📋 Configuration",
            border_style="cyan",
        )
    )

    display_config_summary(config)

    if is_sweep:
        from srtctl.core.sweep import generate_sweep_configs

        configs = generate_sweep_configs(config)
        console.print()
        display_sweep_table(configs)
        console.print(f"\n[bold]Total jobs:[/] [cyan]{len(configs)}[/]")

    console.print()
    return questionary.confirm(
        "Submit to SLURM?",
        default=False,
        style=STYLE,
    ).ask()


def preview_sbatch(config_path: Path, config: dict) -> None:
    """Preview the generated sbatch script."""
    from srtctl.cli.submit import generate_minimal_sbatch_script
    from srtctl.core.config import load_config

    typed_config = load_config(config_path)
    script = generate_minimal_sbatch_script(typed_config, config_path)
    display_sbatch_script(script)


def modify_config_interactive(config: dict) -> dict:
    """Allow interactive modification of config parameters."""
    console.print("\n[bold]Modify Configuration[/]")
    console.print("[dim]Press Enter to keep current value, or type new value[/]\n")

    modified = config.copy()

    # Modifiable fields
    modifiable = [
        ("name", "Job name"),
        ("resources.prefill_workers", "Prefill workers"),
        ("resources.decode_workers", "Decode workers"),
        ("benchmark.isl", "Input sequence length"),
        ("benchmark.osl", "Output sequence length"),
    ]

    for path, label in modifiable:
        parts = path.split(".")
        current = modified
        for p in parts[:-1]:
            current = current.get(p, {})

        current_val = current.get(parts[-1], "N/A")

        new_val = questionary.text(
            f"{label} [{current_val}]:",
            default="",
            style=STYLE,
        ).ask()

        if new_val and new_val.strip():
            # Try to parse as int/float if possible
            try:
                new_val = int(new_val)
            except ValueError:
                with contextlib.suppress(ValueError):
                    new_val = float(new_val)

            # Navigate and set
            target = modified
            for p in parts[:-1]:
                if p not in target:
                    target[p] = {}
                target = target[p]
            target[parts[-1]] = new_val

    return modified


def run_interactive() -> int:
    """Main interactive mode entry point."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]srtctl[/] [dim]Interactive Mode[/]",
            border_style="cyan",
        )
    )
    console.print()

    # Select config
    config_path = select_recipe()
    if not config_path:
        console.print("[yellow]No config selected. Exiting.[/]")
        return 0

    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/]")
        return 1

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    is_sweep = "sweep" in config

    # Show summary
    console.print()
    display_config_summary(config, title=str(config_path))

    # Action menu
    while True:
        console.print()
        action = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("🚀 Submit job(s)", value="submit"),
                questionary.Choice("👁️  Preview sbatch script", value="preview"),
                questionary.Choice("✏️  Modify parameters", value="modify"),
                questionary.Choice("🔍 Dry-run", value="dry-run"),
                questionary.Choice("📁 Select different config", value="reselect"),
                questionary.Choice("❌ Exit", value="exit"),
            ],
            style=STYLE,
        ).ask()

        if action == "exit" or action is None:
            console.print("[dim]Goodbye![/]")
            return 0

        elif action == "reselect":
            config_path = select_recipe()
            if not config_path:
                continue
            with open(config_path) as f:
                config = yaml.safe_load(f)
            is_sweep = "sweep" in config
            display_config_summary(config, title=str(config_path))

        elif action == "preview":
            preview_sbatch(config_path, config)

        elif action == "modify":
            config = modify_config_interactive(config)
            # Save to temp file for submission
            import tempfile

            fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="srtctl_modified_")
            with open(temp_path, "w") as f:
                yaml.dump(config, f)
            config_path = Path(temp_path)
            console.print("[green]Configuration modified.[/]")
            display_config_summary(config)

        elif action == "dry-run":
            from srtctl.cli.submit import submit_single, submit_sweep

            console.print()
            if is_sweep:
                submit_sweep(config_path, dry_run=True)
            else:
                submit_single(config_path=config_path, dry_run=True)

        elif action == "submit":
            if not confirm_submission(config_path, config, is_sweep):
                console.print("[yellow]Submission cancelled.[/]")
                continue

            from srtctl.cli.submit import submit_single, submit_sweep

            try:
                if is_sweep:
                    submit_sweep(config_path, dry_run=False)
                else:
                    submit_single(config_path=config_path, dry_run=False)
                console.print("\n[bold green]✅ Submission complete![/]")
                return 0
            except Exception as e:
                console.print(f"\n[bold red]❌ Submission failed: {e}[/]")
                return 1

    return 0


def main():
    """Entry point for interactive mode."""
    import sys

    sys.exit(run_interactive())


if __name__ == "__main__":
    main()
