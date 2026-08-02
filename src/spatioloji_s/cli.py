"""Console script for spatioloji_s."""

from __future__ import annotations

import importlib.metadata
import importlib.util

import typer
from rich.console import Console
from rich.markup import escape

app = typer.Typer(
    help="Command-line utilities for spatioloji_s.",
    no_args_is_help=True,
)
console = Console()

# Optional feature groups: label -> (import name that gates it, extra that
# installs it). Kept in sync with [project.optional-dependencies] in
# pyproject.toml.
_FEATURES: dict[str, tuple[str, str]] = {
    "Leiden clustering": ("leidenalg", "clustering"),
    "UMAP": ("umap", "reduction"),
    "Batch correction (Harmony)": ("harmonypy", "batch"),
    "Differential expression (DESeq2)": ("pydeseq2", "deg"),
    "AnnData / scanpy interop": ("anndata", "anndata"),
    "Cell type annotation (CellTypist)": ("celltypist", "annotation"),
    "Ripley's K/L": ("pointpats", "ripley"),
    "Pathway scoring": ("decoupler", "decoupler"),
    "Imputation (scVI)": ("scvi", "imputation"),
}


def _installed_version() -> str:
    """Return the installed distribution version.

    Returns:
        The version string from package metadata, or a placeholder when the
        distribution metadata cannot be found (for example when running from a
        source tree that was never installed).
    """
    try:
        return importlib.metadata.version("spatioloji_s")
    except importlib.metadata.PackageNotFoundError:
        return "unknown (not installed)"


def _is_available(module: str) -> bool:
    """Report whether an optional module is importable, without importing it.

    Args:
        module: Top-level module name to look for.

    Returns:
        True if the module can be imported.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@app.command()
def version() -> None:
    """Print the installed spatioloji_s version."""
    console.print(_installed_version())


@app.command()
def info() -> None:
    """Show the installed version and which optional features are available."""
    console.print(f"[bold]spatioloji_s[/bold] {_installed_version()}")
    console.print("\n[bold]Optional features[/bold]")
    for label, (module, extra) in _FEATURES.items():
        if _is_available(module):
            console.print(f"  [green]available[/green]  {label}")
        else:
            # escape() so Rich does not swallow the "[extra]" as a style tag.
            hint = escape(f"pip install 'spatioloji_s[{extra}]'")
            console.print(f"  [yellow]missing[/yellow]    {label}  ->  {hint}")


if __name__ == "__main__":
    app()
