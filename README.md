# Article Briefing

This first project slice reads article text files from a landing folder, uses a
locally hosted Ollama model to create one summary per article plus a briefing
overview, then archives the processed article files. It does not send email yet.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the local project:

   ```bash
   python -m pip install --editable .
   ```

3. Create your local configuration and set `OLLAMA_MODEL` to a model that you
   have already pulled in Ollama:

   ```bash
   cp .env.example .env
   ```

4. Ensure Ollama is running and the configured model is available:

   ```bash
   ollama list
   ```

## Hacker News top stories

Print current Hacker News front-page stories without invoking Ollama or writing
any files:

```bash
hacker-news-top
```

The command defaults to 10 stories. Pass a limit to inspect more:

```bash
hacker-news-top --limit 20
```

Each result includes its Hacker News rank, points, comment count, author, and
both the original linked article URL (when present) and the Hacker News
discussion URL. Hacker News `topstories` are the current ranked front page, not
a fixed calendar-day archive.

## Landing-file format

Place one or more UTF-8 `.txt` files in `data/landing/`. An original article
URL is optional. When available, use either a `URL:` header or a complete
`http`/`https` URL as the first non-empty line. The filename becomes the
article title.

```text
URL: https://example.com/articles/new-release

Article text begins here.
It can span multiple lines.
```

Files with no URL are also valid:

```text
Article text begins here.
It can span multiple lines.
```

An invalid `URL:` header is reported as a warning and left in the landing
folder for correction. Individual article summaries show “Original article:
unavailable” when no URL is provided.

For example, a file named `new-model-release.txt` is displayed as “new model
release” in the generated briefing.

## Run

```bash
python -m article_briefing
```

On success, the command:

1. creates one structured Ollama summary per landing file;
2. writes each individual article summary to
   `data/briefings/<timestamp>/article-summaries/<article-file-name>.md`;
3. creates a short briefing overview from those item summaries, selecting only
   the 2–3 most important cross-article developments;
4. writes JSON and Markdown briefing files in
   `data/briefings/<timestamp>/`; and
5. moves the original `.txt` files to `data/archive/<timestamp>/`.

`briefing.md` contains only the short combined briefing. Open the individual
files in `article-summaries/` to review each article's complete summary, key
points, relevance, and read recommendation.

Malformed landing files do not stop valid files from being processed. The
command prints a warning, includes the warning in `briefing.json`, and leaves
each malformed file in `data/landing/` for correction. Only successfully
summarized files are archived. Errors from Ollama or briefing-output creation
still stop the batch without archiving its valid landing files.

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | Yes | Model passed to Ollama for every generation |
| `OLLAMA_BASE_URL` | No | Defaults to `http://127.0.0.1:11434` |
| `OLLAMA_TIMEOUT_SECONDS` | No | Request timeout, default `180` |
| `LANDING_DIR` | No | Input directory, default `data/landing` |
| `ARCHIVE_DIR` | No | Archived input directory, default `data/archive` |
| `BRIEFING_DIR` | No | Generated briefing directory, default `data/briefings` |
| `MAX_ARTICLE_CHARACTERS` | No | Per-article input cap, default `24000` |

Do not commit `.env`; it may later contain SMTP credentials.
`.env.example` contains only safe placeholders and should be committed so people
cloning the project can create their own local configuration.