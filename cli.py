"""CLI entry point."""
from __future__ import annotations

import asyncio
from pathlib import Path

import click
from loguru import logger


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    if not verbose:
        logger.remove()
        logger.add(lambda msg: click.echo(msg, err=True), level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


@cli.command()
@click.option("--instruction", "-i", required=True, type=click.Path(exists=True), help="Task instruction file")
@click.option("--output", "-o", default="reports", help="Output directory for reports")
@click.option("--personas", "-p", default=None, help="Comma-separated persona archetypes (default: all 7)")
def run(instruction: str, output: str, personas: str | None):
    """Run full evaluation pipeline."""
    from run_eval import run_full_pipeline
    from core.models import PersonaArchetype

    instruction_text = Path(instruction).read_text(encoding="utf-8")
    archetypes = None
    if personas:
        archetypes = [PersonaArchetype(p.strip()) for p in personas.split(",")]

    result = asyncio.run(run_full_pipeline(instruction_text, output, archetypes))
    click.echo(f"\n✅ Evaluation complete!")
    click.echo(f"   Overall score: {result['overall_score']:.1f} / 100")
    click.echo(f"   HTML report:   {result['paths']['html']}")
    click.echo(f"   JSON data:     {result['paths']['json']}")


@cli.command()
@click.option("--instruction", "-i", required=True, type=click.Path(exists=True), help="Task instruction file")
def parse(instruction: str):
    """Parse and display instruction structure."""
    import json
    from config.settings import settings
    from core.llm_client import LLMClient
    from parser.instruction_parser import InstructionParser

    async def _parse():
        client = LLMClient(
            api_key=settings.EVAL_LLM_API_KEY,
            base_url=settings.EVAL_LLM_BASE_URL,
            model=settings.EVAL_LLM_MODEL,
        )
        p = InstructionParser(client)
        text = Path(instruction).read_text(encoding="utf-8")
        spec = await p.parse(text)
        return spec

    spec = asyncio.run(_parse())
    click.echo(f"\n📋 Instruction Spec:")
    click.echo(f"   Role: {spec.role}")
    click.echo(f"   Objective: {spec.objective}")
    click.echo(f"   Flow nodes: {len(spec.flow_graph)}")
    click.echo(f"   Constraints: {len(spec.constraints)}")
    click.echo(f"   Knowledge points: {len(spec.knowledge_points)}")
    click.echo("\n--- Flow Graph ---")
    for node in spec.flow_graph:
        req = "✅" if node.required else "⚪"
        click.echo(f"  {req} [{node.id}] {node.description}")
    click.echo("\n--- Constraints ---")
    for c in spec.constraints:
        icon = "🔴" if c.type.value == "hard" else "🟡"
        click.echo(f"  {icon} [{c.type.value}] {c.description}")
    click.echo("\n--- Knowledge Points ---")
    for k in spec.knowledge_points:
        click.echo(f"  📚 {k.description}")


@cli.command()
@click.option("--instruction", "-i", required=True, type=click.Path(exists=True), help="Task instruction file")
@click.option("--persona", "-p", default="cooperative", help="Persona archetype to simulate")
@click.option("--max-turns", default=20, help="Maximum turns")
def simulate(instruction: str, persona: str, max_turns: int):
    """Simulate a dialogue without evaluation."""
    from config.settings import settings
    from core.llm_client import LLMClient
    from core.models import PersonaArchetype
    from parser.instruction_parser import InstructionParser
    from simulator.engine import DialogueEngine
    from simulator.persona import get_persona

    async def _simulate():
        eval_client = LLMClient(
            api_key=settings.EVAL_LLM_API_KEY,
            base_url=settings.EVAL_LLM_BASE_URL,
            model=settings.EVAL_LLM_MODEL,
        )
        agent_client = LLMClient(
            api_key=settings.AGENT_LLM_API_KEY,
            base_url=settings.AGENT_LLM_BASE_URL,
            model=settings.AGENT_LLM_MODEL,
        )
        sim_client = LLMClient(
            api_key=settings.SIM_LLM_API_KEY,
            base_url=settings.SIM_LLM_BASE_URL,
            model=settings.SIM_LLM_MODEL,
        )
        text = Path(instruction).read_text(encoding="utf-8")
        parser = InstructionParser(eval_client)
        spec = await parser.parse(text)

        p = get_persona(PersonaArchetype(persona))
        engine = DialogueEngine(agent_client, sim_client, max_turns=max_turns)
        dialogue = await engine.run(spec, p)
        return dialogue

    dialogue = asyncio.run(_simulate())
    click.echo(f"\n💬 Simulated Dialogue ({dialogue.persona.archetype.value})")
    click.echo(f"   Turns: {len(dialogue.turns)}")
    click.echo(f"   Termination: {dialogue.termination_reason.value}")
    click.echo("\n" + "=" * 60)
    for turn in dialogue.turns:
        role = "🤖 Agent" if turn.role == "agent" else "👤 User "
        click.echo(f"\n[{turn.turn_id:2d}] {role} ({turn.char_count}字):")
        click.echo(f"     {turn.content}")


@cli.command()
@click.option("--port", default=8000, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--dev", is_flag=True, help="Also start Vite dev server on :5173 for hot-reload")
def serve(port: int, host: str, dev: bool):
    """Start the FastAPI web server.

    For production: ensure the frontend is built first:
      cd web && npm install && npm run build
    """
    import subprocess
    import uvicorn
    from api.app import app

    web_dist = Path(__file__).parent / "web" / "dist"
    if not web_dist.exists():
        click.secho(
            "⚠  web/dist/ not found. Run `cd web && npm install && npm run build` first, "
            "or use --dev to start the Vite dev server alongside.",
            fg="yellow",
        )

    if dev:
        import threading
        vite_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(Path(__file__).parent / "web"),
        )
        click.echo("🎨 Vite dev server started at http://localhost:5173 (hot-reload)")

    click.echo(f"🚀 Starting API server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
