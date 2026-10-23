# grokbot

Talk to Claude Code out loud and hear it talk back, hands-free.

You speak, grokbot transcribes you on your own machine, Claude Code does the work in your project, and a local voice reads the reply. You can talk over it to cut in, answer its permission questions by voice, and pick the conversation up later in plain `claude`.

- **Runs on your Claude plan.** grokbot drives the regular `claude` CLI signed in with your claude.ai account. It refuses to start if Claude Code isn't on a plan, and it strips `ANTHROPIC_API_KEY` and similar variables so nothing gets billed to an API account.
- **Speech never leaves your computer.** Speech recognition (faster-whisper) and the voice (Kokoro) run locally, so they cost nothing and don't touch your plan's usage.

## Start

Open a terminal in the project you want to work on and run:

```
grokbot
```

Wait for the greeting, then just talk. Pause for a moment and grokbot answers. The first run downloads the voice model (about 350 MB).

| Key | Does |
|---|---|
| just type | Your words go on the input line at the bottom; `Enter` sends. Pasting keeps every line, and `Ctrl+Enter` adds one. The mic waits while there's text on the line. |
| `Esc` (or `space` on an empty line) | Stop talking and cancel the current reply. With nothing to stop, `Esc` clears what you typed. |
| `shift+tab` | Switch modes: ask first → auto-edit → auto (see below) |
| `y` / `n` / `a` | Answer a permission question: yes / no / always (on an empty line) |
| `/mute` | Mute or unmute the mic |
| `/repeat` | Say the last reply again |
| `/faster` / `/slower` | Talk faster or slower |
| `/quit` or `Ctrl+C` | Quit (prints how to resume the conversation) |
| `/help` | Show these |

**Talking over it:** start speaking while grokbot talks and it pauses to listen. If it was just hearing its own voice through your speakers, it carries on. After that happens twice it stops listening while it talks, and you cut in with `Esc` instead. Headphones avoid the problem.

**Permissions:** when Claude wants to edit a file or run a command, grokbot asks, for example "Can I edit app.py?". You can answer:

- "yes" or "go ahead" to allow it once.
- "always" to allow that kind of action for the rest of the session.
- "auto" or "auto mode" to allow it and switch to auto mode (below).
- "no", or anything else ("no, put it in utils instead"), to deny it. Claude hears what you said and adjusts.

**Auto mode:** tired of yes and no? Say "auto mode" (or press `shift+tab` until the badge on the status line says AUTO, or start with `grokbot --auto`). Claude then edits files and runs commands without asking. Claude Code's own safety check still runs in auto mode: it blocks risky actions, such as force-pushing, deleting things it can't account for, or sending secrets out, and tells Claude to find another way. Anything that still needs your OK comes to you as a question. Say "auto mode off" or "ask me first" to go back. In between is auto-edit ("accept edits"): file edits go ahead, and commands still ask.

| Mode | Badge | Asks before |
|---|---|---|
| ask first (the default) | `ask` | edits and commands |
| auto-edit | `auto-edit` | commands |
| auto | `AUTO` | only what Claude Code's safety check flags |

**Seeing what it's doing:** while Claude works, the status line shows how long it's been going, how many steps it has taken and the current one, for example `0:23 · 4 steps · editing app.py`, or `1,200 tokens of thought` while it thinks. Each step is listed as it happens, a step that fails gets a red ✗ line with the error, and a long reply ends with a summary like `done in 1:10 · 6 steps`. The reply text appears on screen as it's spoken. If Claude has been working quietly for 20 seconds, grokbot says what it's up to ("Still working. Running the tests."). Talking over one of these check-ins doesn't cancel the work; what you say is added to it.

**Other things you can say:** "say that again" or "repeat that" replays the last reply, read locally without using your plan. These commands, and the mode switches, only count when they're the whole sentence, so "turn on auto mode and fix the tests" goes to Claude as a normal request.

**Picking up later:** when you quit, grokbot prints the session id. Continue by voice with `grokbot --resume <id>`, by text with `claude --resume <id>`, or with `grokbot -c` for the latest conversation in that folder.

## Options

```
grokbot --model sonnet        # snappier replies than Opus; haiku is quicker still
grokbot --effort low          # less thinking before answering
grokbot --permission-mode acceptEdits   # edit files without asking (still asks for commands)
grokbot --auto                # start in auto mode: no yes/no questions
grokbot --no-progress         # no spoken "still working" check-ins
grokbot --voice am_michael --speed 1.15
grokbot --mic GHW --speaker GHW         # any part of a device name, or its number
grokbot --pause 1.0           # wait longer before deciding you've finished
grokbot --no-barge-in         # never listen while speaking
```

| Command | Does |
|---|---|
| `grokbot devices` | List microphones and speakers |
| `grokbot voices` | List voices |
| `grokbot say "Hey there" --voice bf_emma` | Try a voice |
| `grokbot doctor` | Check the plan, models, GPU, mic, and speakers |
| `grokbot setup` | Download models ahead of time |

To keep settings, put them in `%APPDATA%\grokbot\config.toml`:

```toml
model = "sonnet"
permission_mode = "auto"   # or "manual" (ask first, the default), "acceptEdits", "plan"
voice = "af_heart"
speed = 1.1
mic = "GHW"
speaker = "GHW"
end_silence = 0.8     # seconds of quiet that end your turn
barge_in = true
chime = true          # a soft blip when it has heard you
progress_voice = true # say what it's doing during long quiet stretches
progress_every = 20   # seconds of quiet before each of those
stt_hints = true      # tell speech recognition your project's file names
```

The personality lives in `src/grokbot/persona.md`. Edit it, or pass `--persona my-persona.md`.

## Using it from Neovim

Run it in Neovim's terminal so you can watch files change while you talk:

1. `:cd` to your project (or open Neovim there).
2. Press `Ctrl-/` to open the terminal, then type `grokbot`.
3. Press `Ctrl-k` to jump up to your code while it talks; voice keeps working. Press `Ctrl-j` to come back when you want to type. If keys don't reach it, press `i` first.
4. Press `Ctrl-/` to hide the terminal. grokbot keeps listening in the background.

## How it works

```
mic ─► Silero VAD ─► faster-whisper ─► claude -p (stream-json, your plan) ─► sentence splitter ─► Kokoro ─► speakers
         (when you talk)  (GPU or CPU)      tools, edits, permission prompts     (skips code, URLs)
```

- `app.py`: the conversation loop and turn-taking rules (one thread, fed by an event queue).
- `claude.py`: runs Claude Code, parses its stream (text, thinking, steps and their results, mode changes), answers permission prompts, switches modes.
- `progress.py`: what Claude is doing right now, for the status line, the turn summary and the spoken check-ins.
- `vad.py`, `stt.py`, `vocab.py`: hearing you. `tts.py`, `audio.py`, `speech_text.py`: talking back.
- `intent.py`: yes/no/always/auto, "stop", "auto mode", "say that again", and "was that just my own echo?".

Speech recognition uses `large-v3-turbo` on an NVIDIA GPU, about 0.2 s per sentence, through the optional `gpu` extra (CUDA libraries from pip). Without a GPU it uses `base.en` on the CPU. It's given a list of words to expect: your project's file and folder names, plus names Claude has mentioned or touched this session. That's why it writes `config.toml` rather than "config tommel".

Claude Code starts while the speech models load (the models load side by side), and the first sentence of each reply is kept short so the voice starts sooner.

## Development

```
uv sync --extra gpu
uv run pytest                       # unit tests
uv run python scripts/simulate.py   # full rehearsal with fake audio against real Claude Code (uses a little of your plan, on Haiku)
```

The `grokbot` command is an editable install (`uv tool install --editable ".[gpu]" --python 3.12`), so code changes apply immediately. Re-run that command after changing dependencies.

## Troubleshooting

- **It doesn't hear me.** Run `grokbot doctor` and check the mic's peak level while you talk. Pick the right mic with `--mic`, and allow microphone access for desktop apps in Windows Settings → Privacy & security → Microphone.
- **It cuts me off mid-thought.** Raise `--pause` (try 1.0 or 1.2).
- **It keeps interrupting itself.** Your mic hears the speakers. Use headphones or `--no-barge-in`.
- **It misspells names.** It already knows your project's file names. For anything else, just type it. If it ever makes up words when you didn't say anything, set `stt_hints = false`.
- **The check-ins are too chatty.** Raise `progress_every` (say 45), or turn them off with `--no-progress`.
- **"isn't on a Claude plan".** Run `claude`, type `/login`, and choose your Claude subscription.
- **Replies feel slow.** Try `--model sonnet` or `--effort low`. Most of the wait is Claude thinking; hearing you and speaking take well under a second.
