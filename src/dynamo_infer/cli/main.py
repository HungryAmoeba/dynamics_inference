"""Main CLI entry point for dynamo-infer."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

from ..workflow import run_pipeline_from_config_file

app = typer.Typer(
    name="dynamo-infer",
    help="Modular dynamics simulation and inference framework",
    add_completion=False,
)

console = Console()


@app.command()
def main(
    config: Path = typer.Argument(..., help="Configuration file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
    save_intermediate: bool = typer.Option(True, "--save/--no-save", help="Save intermediate results"),
):
    """Run the complete dynamo-infer pipeline."""
    try:
        results = run_pipeline_from_config_file(
            str(config),
            output_dir=str(output) if output else None,
            verbose=verbose,
            save_intermediate=save_intermediate,
        )
        
        if verbose:
            console.print(f"[green]✅ Pipeline completed successfully![/green]")
            console.print(f"Results available in: {results.get('output_dir', 'outputs/')}")
            
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()