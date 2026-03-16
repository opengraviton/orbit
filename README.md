# Orbit — Local AI Agent

**Autonomous AI that runs on your machine.** Code execution, browser control, web search, model training, deployment.

## Vision

An efficient local superintelligence that can:
- Write and execute code
- Fetch any data from the internet (URLs, APIs, files)
- Control the browser, fill forms, create accounts
- Train small "slave" models for specialized tasks
- Deploy models and call them (local or HuggingFace)
- Self-improve: generate data → train → deploy → use

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (Graviton / Omega / local LLM)                 │
│  Planning, reasoning, tool selection                        │
└─────────────────────┬─────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    ▼                 ▼                  ▼
┌─────────┐    ┌─────────────┐    ┌──────────────┐
│ code    │    │ browser     │    │ train_model  │
│ exec    │    │ control     │    │ deploy       │
└─────────┘    └─────────────┘    └──────────────┘
```

## Installation

```bash
cd orbit
pip install -e ".[full]"
```

For full browser control (visible window, click, type):
```bash
pip install playwright pyautogui
playwright install chromium
```

On macOS, grant **Accessibility** permission for mouse/keyboard control (System Settings → Privacy & Security).

## Quick Start

```bash
# Run a task (Qwen 7B default)
orbit run --task "Run whoami"
orbit run --task "Search for AI news"

# Smaller model (less RAM)
orbit run --model Qwen/Qwen2.5-3B-Instruct --task "Run ls"

# Full control: browser, terminal, mouse, keyboard
orbit autonomous --full-control
```

## Tools

| Tool | Description |
|------|-------------|
| `code_exec` | Execute Python/shell code |
| `run_command` | Run any shell command — full system access |
| `web_search` | Search the internet |
| `fetch_url` | Fetch any URL — HTML, JSON, or save to file |
| `browser_open` | Open visible browser window |
| `browser_navigate` | Navigate to URL |
| `browser_click` | Click element by selector or text |
| `browser_type` | Type text into page |
| `browser_screenshot` | Take screenshot |
| `browser` | Control Chromium (headless: navigate, screenshot, extract) |
| `browser_form` | Fill and submit forms (signup, login) |
| `mouse_move` | Move mouse to x,y |
| `mouse_click` | Click at position |
| `keyboard_type` | Type text via keyboard |
| `keyboard_press` | Press key (enter, tab, etc.) |
| `screen_size` | Get screen dimensions |
| `train_model` | Train small model via graviton-native |
| `deploy` | Deploy model to HuggingFace |
| `generate_training_data` | Generate training examples with current model |
| `self_train` | Train on custom data |
| `self_optimize` | Optimization guidance |
| `train_deploy` | Create slave model — train + deploy in one step |
| `call_deployed_model` | Call a deployed/slave model with a prompt |
| `call_model` | Call ANY model (OpenAI, Anthropic, HF, local) |
| `self_prompt` | Set your own next goal — autonomous |

## AGI Mode: Autonomous

`orbit autonomous` — Agent runs without user task. It sets its own goals, uses `self_prompt` to continue:

```bash
orbit autonomous  # Agent picks first goal, sets next ones
orbit autonomous --seed-goal "Generate 50 training examples and train a slave model"
```

## Slave Models

Orbit can create specialized sub-models:

1. **fetch_url** — Get training data from the internet
2. **generate_training_data** — Or generate with current model
3. **train_deploy** — Train + deploy to HuggingFace
4. **call_deployed_model** — Use the slave for specific tasks

## Self-Improvement Loop

1. **generate_training_data** → **self_train** → **deploy**
2. Run with new checkpoint: `orbit run --model path/to/improved`

## Requirements

- Python 3.9+
- Graviton or HuggingFace model
- 8GB+ RAM for local inference

## License

Apache-2.0
