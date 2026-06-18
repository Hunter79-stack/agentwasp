# Quickstart

10 minutes from zero to a working WASP instance.

## 1. Install

On a Linux box (VPS or local) with `curl`:

```bash
curl -fsSL https://agentwasp.com/install.sh | bash
```

The installer:
1. Detects your OS (Debian / Ubuntu / RHEL / AlmaLinux / Rocky / Fedora / Arch / openSUSE / Alpine / macOS).
2. Installs Docker if missing.
3. Downloads WASP to `/opt/wasp` (release tarball by default; `--install-method=git` for a clone).
4. Generates `.env` with secure secrets (Postgres password, dashboard secret, media signing secret — all via `openssl rand`).
5. Launches `wasp onboard` to ask you a few questions.
6. Builds the containers (~3 minutes the first time).
7. Starts the stack.
8. Runs health checks and prints the dashboard URL.

## 2. Onboarding answers

The installer will ask:

| Prompt | What to enter |
|---|---|
| Timezone | Your IANA timezone, e.g. `America/Santiago`, `Europe/Berlin` |
| Telegram bot token | From [@BotFather](https://t.me/BotFather) — or leave blank to disable Telegram entirely |
| Your Telegram user ID | **Required if you set a bot token.** Numeric 5–15 digits (get it from [@userinfobot](https://t.me/userinfobot)). The wizard replicates it to both `TELEGRAM_ALLOWED_USERS` and `SCHEDULER_NOTIFY_CHAT_ID`. There is no public-bot mode. |
| Default LLM provider | `anthropic`, `openai`, `openai-codex`, `xai`, or `google` (or `ollama` for fully self-hosted) |
| Provider API key / OAuth token | Your key for the provider above. `openai-codex` is experimental OAuth bearer-token mode; use `OPENAI_API_KEY`/`openai` for normal OpenAI API billing. |
| Dashboard username | Anything (default: `admin`) |
| Dashboard password | A strong password — saved to `.env`, used to log in |

You can re-run `wasp onboard` any time to change values.

If you choose `openai-codex`, configure only the OAuth access token and optional expiry metadata; do not paste browser cookies, refresh tokens, or full session dumps into onboarding, chat, or GitHub support threads.

## 3. Open the dashboard

After install, open `http://<your-host>:8080` in your browser. Log in with the dashboard credentials you just set.

What you'll see:
- **Chat tab**: talk to the agent.
- **Tasks tab**: scheduled jobs the agent runs in the background.
- **Memory tab**: what the agent remembers, with controls to edit/delete.
- **Goals tab**: long-running plans the agent is executing.
- **Cognitive tab**: internal state — epistemic confidence, CPI, integrity, dreams.
- **Skills tab**: registered skills, with toggle/edit/delete.

## 4. First conversation

Try one of these:

- "What's the weather in <your city>?"
- "Remind me to drink water in 30 minutes"
- "Take a screenshot of example.com"
- "Search the web for 'latest AI news'"

The agent picks the right skill, executes it, and responds. Check the **Tasks** tab to see the reminder you just created.

## 5. Telegram (optional)

If you set a Telegram token during onboarding, just message your bot — same agent, same memory, same skills. The dashboard and Telegram are two front-ends for the same backend.

## 6. Stop / start / update

```bash
wasp stop          # stop everything
wasp start         # start again
wasp restart       # restart in place
wasp update        # pull latest source, rebuild, restart
wasp backup        # save state to ./backups/wasp-<timestamp>.tar.gz
wasp health        # is everything alive?
```

## What's next

- [INSTALL.md](INSTALL.md) — full env reference, advanced options
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — when something breaks
- [SECURITY.md](SECURITY.md) — before you expose this to the public internet
