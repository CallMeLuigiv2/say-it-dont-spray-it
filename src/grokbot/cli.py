"""Command line: `grokbot` starts talking; the other commands set things up and check them."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from rich.console import Console

from grokbot import __version__

COMMANDS = ("talk", "setup", "devices", "voices", "say", "doctor")
PLANS = {"max": "Claude Max", "pro": "Claude Pro", "team": "Claude Team", "enterprise": "Claude Enterprise"}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in (*COMMANDS, "-h", "--help", "--version"):
        args.insert(0, "talk")
    parsed = _parser().parse_args(args)
    for stream in (sys.stdout, sys.stderr):
        if stream and (stream.encoding or "").lower() not in ("utf-8", "utf8"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    console = Console(highlight=False)
    try:
        return parsed.run(parsed, console) or 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # show a clean one-line error instead of a traceback
        from grokbot import audio, claude, tts

        if isinstance(exc, (claude.PlanError, audio.DeviceError, tts.ModelsMissing, ValueError, RuntimeError)):
            console.print(f"[bold red]{exc}[/]")
            return 1
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grokbot", description="Talk to Claude Code out loud. Runs on your Claude plan, not the API.",
    )
    parser.add_argument("--version", action="version", version=f"grokbot {__version__}")
    commands = parser.add_subparsers(metavar="command")

    talk = commands.add_parser("talk", help="have a voice conversation (the default)")
    talk.add_argument("-c", "--continue", dest="continue_session", action="store_true", default=None,
                      help="pick up the most recent conversation in this folder")
    talk.add_argument("-r", "--resume", metavar="SESSION_ID", help="pick up a specific conversation")
    talk.add_argument("--cwd", type=Path, help="project folder to work in (default: the current folder)")
    talk.add_argument("--model", help="Claude model: sonnet, opus, haiku, … (default: your Claude Code default)")
    talk.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="thinking effort")
    talk.add_argument("--permission-mode", choices=["manual", "acceptEdits", "auto", "plan"],
                      help="what Claude may do without asking (default: ask for edits and commands)")
    talk.add_argument("--auto", dest="permission_mode", action="store_const", const="auto",
                      help="start in auto mode: no yes/no questions (same as --permission-mode auto)")
    talk.add_argument("--progress", dest="progress_voice", action=argparse.BooleanOptionalAction, default=None,
                      help="say what I'm doing during long quiet stretches of work (default: on)")
    _voice_args(talk)
    talk.add_argument("--mic", help="microphone: part of its name or its number (see `grokbot devices`)")
    talk.add_argument("--whisper-model",
                      help="speech recognition model (default: large-v3-turbo on a GPU, base.en on a CPU)")
    talk.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"], help="where speech recognition runs")
    talk.add_argument("--pause", type=float, metavar="SECONDS",
                      help="how long you pause before I answer (default: 0.7)")
    talk.add_argument("--barge-in", action=argparse.BooleanOptionalAction, default=None,
                      help="let you talk over me to cut in (default: on)")
    talk.add_argument("--chime", action=argparse.BooleanOptionalAction, default=None,
                      help="blip when I've heard you (default: on)")
    talk.add_argument("--persona", type=Path, help="markdown file that replaces the built-in voice persona")
    talk.set_defaults(run=cmd_talk)

    setup = commands.add_parser("setup", help="download the voice and speech recognition models")
    setup.add_argument("--whisper-model",
                       help="speech recognition model to fetch (default: the one that suits your hardware)")
    setup.set_defaults(run=cmd_setup)

    commands.add_parser("devices", help="list microphones and speakers").set_defaults(run=cmd_devices)
    commands.add_parser("voices", help="list the voices").set_defaults(run=cmd_voices)

    say = commands.add_parser("say", help="say something, to try a voice")
    say.add_argument("text", nargs="+")
    _voice_args(say)
    say.set_defaults(run=cmd_say)

    commands.add_parser("doctor", help="check that everything works").set_defaults(run=cmd_doctor)
    return parser


def _voice_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voice", help="voice name (see `grokbot voices`; default: af_heart)")
    parser.add_argument("--speed", type=float, help="speaking speed (default: 1.05)")
    parser.add_argument("--speaker", help="speakers: part of the name or the number (see `grokbot devices`)")


def _settings(args: argparse.Namespace):
    from grokbot.config import load_settings

    names = ("cwd", "model", "effort", "permission_mode", "resume", "continue_session", "persona", "voice",
             "speed", "mic", "speaker", "whisper_model", "whisper_device", "barge_in", "chime", "progress_voice")
    overrides = {name: getattr(args, name, None) for name in names}
    overrides["end_silence"] = getattr(args, "pause", None)
    return load_settings(overrides)


# ---- commands -------------------------------------------------------------------------------


def cmd_talk(args: argparse.Namespace, console: Console) -> int:
    from grokbot import audio, claude, models
    from grokbot.app import App
    from grokbot.stt import default_model, pick_device
    from grokbot.ui import MODES, UI

    settings = _settings(args)
    status = claude.check_plan(claude.find_claude())
    if not models.kokoro_ready():
        console.print("First run: downloading the voice model (about 350 MB)…")
        models.download_kokoro(console)
    whisper = settings.whisper_model or default_model(pick_device(settings.whisper_device))
    if not models.whisper_ready(whisper):
        console.print(f"Downloading the speech recognition model {whisper} (one time; large-v3-turbo is about 1.6 GB)…")
        models.download_whisper(whisper)
    app = App(settings, UI(console))
    with console.status("Loading speech recognition and the voice…"):
        app.load()

    plan = PLANS.get(str(status.get("subscriptionType")), "your Claude plan")
    hardware = "GPU" if app.stt.device == "cuda" else "CPU"
    console.print(f"[bold magenta]grokbot[/] · Claude Code on {plan}, no API billing")
    console.print(f"[dim]folder [/]{settings.cwd}")
    console.print(f"[dim]mic    [/]{audio.device_name(app.mic_device, 'input')}"
                  f"[dim]   speaker [/]{audio.device_name(app.speaker_device, 'output')}")
    console.print(f"[dim]voice  [/]{settings.voice}[dim]   hearing [/]{app.stt.model_name} on {hardware}")
    badge, _style, meaning = MODES.get(app.mode, (app.mode, "", ""))
    console.print(f"[dim]mode   [/]{badge}[dim]   {meaning}. Press shift+tab or say \"auto mode\" to change.[/]")
    talk_over = "Talk over me to cut in." if settings.barge_in else "Press Esc to cut me off."
    console.print(f"[green]Just talk, or type. Pause and I'll answer. {talk_over}[/]\n")
    try:
        app.run()
    finally:
        if app.session_id:
            console.print(
                f"\n[dim]Saved. Pick this conversation back up with [bold]grokbot --resume {app.session_id}[/], "
                f"or in text with [bold]claude --resume {app.session_id}[/].[/]"
            )
    return 0


def cmd_setup(args: argparse.Namespace, console: Console) -> int:
    from grokbot import models
    from grokbot.config import load_settings
    from grokbot.stt import default_model, pick_device

    settings = load_settings()
    whisper = args.whisper_model or settings.whisper_model or default_model(pick_device(settings.whisper_device))
    models.download_kokoro(console)
    with console.status(f"Fetching speech recognition model {whisper}…"):
        models.download_whisper(whisper)
    console.print("[green]Models ready.[/] Run [bold]grokbot doctor[/] to check your mic and speakers.")
    return 0


def cmd_devices(args: argparse.Namespace, console: Console) -> int:
    from grokbot import audio

    for kind, title in (("input", "Microphones"), ("output", "Speakers")):
        console.print(f"[bold]{title}[/]")
        for index, name, is_default in audio.list_devices(kind):
            mark = "[green]← default[/]" if is_default else ""
            console.print(f"  {index:>3}  {name} {mark}")
    console.print('\nUse one with [bold]grokbot --mic "GHW"[/] (any part of the name, or the number), '
                  "or set mic and speaker in the config file.")
    return 0


def cmd_voices(args: argparse.Namespace, console: Console) -> int:
    from grokbot.tts import VOICE_GROUPS, Voice

    voice = Voice()
    for prefix, label in VOICE_GROUPS.items():
        names = [n for n in voice.names if n.startswith(prefix)]
        console.print(f"[bold]{label}[/]  " + "  ".join(names))
    console.print('\nTry one with [bold]grokbot say "Hey, what are we building?" --voice am_michael[/]')
    return 0


def cmd_say(args: argparse.Namespace, console: Console) -> int:
    from grokbot import audio
    from grokbot.tts import Voice

    settings = _settings(args)
    voice = Voice(settings.voice, settings.speed)
    done = threading.Event()
    player = audio.Player(audio.resolve_device(settings.speaker, "output"), on_idle=done.set)
    player.start()
    player.play(voice.synth(" ".join(args.text)))
    done.wait(timeout=120)
    time.sleep(0.3)  # let the device finish its buffer
    player.close()
    return 0


def cmd_doctor(args: argparse.Namespace, console: Console) -> int:
    import os

    import numpy as np

    from grokbot import audio, claude, models
    from grokbot.config import load_settings
    from grokbot.paths import config_path

    failures = 0

    def report(ok: bool, label: str, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f" {mark} {label}" + (f" [dim]{detail}[/]" if detail else ""))

    try:
        settings = load_settings()
        report(True, "Settings", str(config_path()) if config_path().exists() else "defaults (no config file)")
    except ValueError as exc:
        report(False, "Settings", str(exc))
        settings = None

    try:
        status = claude.check_plan(claude.find_claude())
        plan = PLANS.get(str(status.get("subscriptionType")), str(status.get("subscriptionType")))
        report(True, "Claude Code is on your plan", plan)
    except claude.PlanError as exc:
        report(False, "Claude Code is on your plan", str(exc))
    stripped = [v for v in claude.API_BILLING_VARS if v in os.environ]
    if stripped:
        report(True, "API key variables are set, but grokbot hides them from Claude Code", ", ".join(stripped))

    report(models.kokoro_ready(), "Voice model downloaded", "" if models.kokoro_ready() else "run `grokbot setup`")

    try:
        from grokbot.stt import Transcriber

        stt = Transcriber(settings.whisper_model if settings else None, settings.whisper_device if settings else "auto")
        report(True, f"Speech recognition ({stt.model_name})", f"on {'GPU' if stt.device == 'cuda' else 'CPU'}")
    except Exception as exc:
        report(False, "Speech recognition", str(exc))
        stt = None

    voice = None
    if models.kokoro_ready():
        from grokbot.tts import Voice

        try:
            voice = Voice(settings.voice if settings else "af_heart")
            report(True, "Voice", voice.name)
        except Exception as exc:
            report(False, "Voice", str(exc))

    if voice and stt:
        phrase = "Open the main file and add a function that reads the settings."
        clip = audio.resample((voice.synth(phrase) * 32767).astype(np.int16), audio.SPEAKER_RATE, audio.MIC_RATE)
        started = time.perf_counter()
        heard = stt.transcribe(clip.astype(np.float32) / 32768)
        took = time.perf_counter() - started
        same = heard.lower().strip(" .") == phrase.lower().strip(" .")
        report(same, "Voice → speech recognition round trip", f"{took:.2f}s, heard {heard!r}")

    try:
        frames: list[np.ndarray] = []
        mic_index = audio.resolve_device(settings.mic if settings else None, "input")
        mic = audio.Mic(mic_index, frames.append)
        mic.start()
        time.sleep(1.0)
        mic.close()
        loudest = max((audio.rms(f) for f in frames), default=0.0)
        ok = len(frames) > 20 and loudest > 0
        report(ok, "Microphone", f"{audio.device_name(mic_index, 'input')}, "
                                 f"{len(frames)} frames in 1s, peak level {loudest:.4f}")
    except Exception as exc:
        report(False, "Microphone", str(exc))

    try:
        speaker_index = audio.resolve_device(settings.speaker if settings else None, "output")
        done = threading.Event()
        player = audio.Player(speaker_index, on_idle=done.set)
        player.start()
        player.play(np.zeros(audio.SPEAKER_RATE // 5, dtype=np.float32))  # 0.2 s of silence
        report(done.wait(timeout=5), "Speakers", f"{audio.device_name(speaker_index, 'output')} (played silence)")
        player.close()
    except Exception as exc:
        report(False, "Speakers", str(exc))

    console.print("\n[green]All good. Run [bold]grokbot[/] and start talking.[/]" if not failures
                  else f"\n[red]{failures} problem(s) above.[/]")
    return 1 if failures else 0
