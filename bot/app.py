"""The Discord bot: /translate with attached images or a zip, result back by DM or in the channel.

A thin shell: it checks the upload and the user's limits, queues the job, and
sends the result. All the work is `mangatr.chapter`. Needs `pip install discord.py`
and the environment: DISCORD_TOKEN, plus the provider key (see mangatr.config).
Run: python -m bot.app
"""

from __future__ import annotations

import functools
import os
import shutil
import tempfile
from pathlib import Path

from .delivery import parts
from .jobs import Job, JobQueue, Status
from .limits import LimitError, Limits
from .safety import Upload, UploadError, extract


@functools.cache
def _heavy(models_dir: str, cpu: bool):
    """The models, loaded once and shared by every job."""
    from mangatr.compare import variants
    from mangatr.detect import Detector
    from mangatr.ocr import OcrStage
    from mangatr.ocr_models import MangaOcr
    from mangatr.runtime import open_session

    models = Path(models_dir)
    chosen, _ = variants(models, cpu=cpu)
    cheap, tier = chosen["cheap+aot" if "cheap+aot" in chosen else "opencv"]
    detector = Detector(open_session(str(models / "detector.onnx"), prefer_gpu=not cpu))
    ocr = OcrStage([MangaOcr.load(device="cpu" if cpu else None)])
    return detector, ocr, cheap, tier


@functools.cache
def _typesetter(language: str):
    from mangatr.typeset import Typesetter

    return Typesetter(language)


def build_runner(models_dir: str, provider: str, model: str | None, language: str, cpu: bool):
    """A runner for one chapter: shared models, a fresh translation conversation."""
    from mangatr.chapter import ChapterRunner
    from mangatr.chapter_translate import ChapterTranslator
    from mangatr.config import make_provider
    from mangatr.languages import get_language

    detector, ocr, cheap, tier = _heavy(models_dir, cpu)
    translator = ChapterTranslator(make_provider(provider, model), get_language(language))
    return ChapterRunner(detector, ocr, cheap, tier, translator, _typesetter(language))


def main() -> None:
    import discord
    from discord import app_commands

    token = os.environ["DISCORD_TOKEN"]
    models_dir = os.environ.get("MANGATR_MODELS", "models")
    provider = os.environ.get("MANGATR_PROVIDER", "gemini")
    cpu = os.environ.get("MANGATR_CPU") == "1"
    limits = Limits()
    work_root = Path(tempfile.mkdtemp(prefix="mangatr-bot-"))

    client = discord.Client(intents=discord.Intents.default())
    tree = app_commands.CommandTree(client)
    notify: dict[int, discord.Interaction] = {}

    def work(job: Job) -> list[Path]:
        folder, language = job.payload
        runner = build_runner(
            models_dir, provider, None, language, cpu
        )  # a fresh chapter each time
        result = runner.run(sorted((folder / "in").iterdir()), folder / "out")
        done = len(result.outputs)
        limits.finish(job.user_id, done, job.pages)
        if not result.outputs:
            raise RuntimeError("no page could be processed")
        return parts(result.outputs, folder / "send")

    async def finish(job: Job) -> None:
        interaction = notify.pop(job.id)
        folder = job.payload[0]
        try:
            if job.status is Status.DONE:
                files = [discord.File(p) for p in job.result]
                await interaction.followup.send("Done.", files=files[:10])
            else:
                limits.finish(job.user_id)
                await interaction.followup.send(f"Failed: {job.error}")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    queue = JobQueue(work, finish)

    @tree.command(name="translate", description="Translate manga pages (images, .zip or .cbz)")
    @app_commands.describe(file="images, a .zip or a .cbz", language="target: ar or en")
    @app_commands.choices(
        language=[
            app_commands.Choice(name="Arabic", value="ar"),
            app_commands.Choice(name="English", value="en"),
        ]
    )
    async def translate(
        interaction: discord.Interaction, file: discord.Attachment, language: str = "ar"
    ):
        await interaction.response.defer(thinking=True, ephemeral=False)
        folder = Path(tempfile.mkdtemp(dir=work_root))
        try:
            data = await file.read()
            pages = extract(
                [Upload(file.filename, data)], folder / "in", max_pages=limits.max_pages_per_job
            )
            limits.admit(interaction.user.id, len(pages))
        except (UploadError, LimitError) as exc:
            shutil.rmtree(folder, ignore_errors=True)
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        try:
            job = queue.submit(interaction.user.id, len(pages), (folder, language))
        except RuntimeError as exc:
            limits.finish(interaction.user.id, 0, len(pages))
            shutil.rmtree(folder, ignore_errors=True)
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        notify[job.id] = interaction
        await interaction.followup.send(
            f"Queued: {len(pages)} pages, position {queue.position(job)}."
        )

    @client.event
    async def on_ready():
        queue.start()
        await tree.sync()

    client.run(token)


if __name__ == "__main__":
    main()
