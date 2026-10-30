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

### Collect Hacker News article text

Fetch the current top stories' public linked pages, extract their readable
article text, and place successful results in `data/landing/`:

```bash
hacker-news-collect --limit 10
```

The collector does not invoke Ollama. Each created landing file begins with the
final article URL and then contains only extracted readable text, ready for:

```bash
python -m article_briefing
```

Stories with no external URL, non-HTML pages, inaccessible pages, duplicate
landing files, or less than 500 extracted characters are skipped and reported
in the terminal. The collector does not bypass login pages, paywalls, CAPTCHAs,
or bot protections.

The collector supports gzip-compressed HTML and limits both downloaded and
decompressed page data to 10 MB. Malformed story URLs, unsupported content or
character encodings, invalid page text, article-extraction failures, and
landing-file write failures are isolated to that story. The collector reports
the failure and continues with the remaining Hacker News stories.

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

While it runs, the command prints each processing stage, attempt and error to
the terminal, along with complete Ollama HTTP response bodies and model
responses, including invalid responses. This can be very verbose and may
expose article-derived or other sensitive information in terminal scrollback
or redirected logs. Request contents and SMTP credentials are not logged.

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

The briefing output includes the overall generation time, total process time,
overview summarization time, and an individual Ollama duration for every
article summary. When email is enabled, it also records SMTP delivery time.

If Ollama returns an invalid individual article summary, the program retries
that article once. If the second attempt also fails, it continues with the
remaining articles, leaves the failed source file in `data/landing/`, and lists
the failure under “Articles not included” in the saved briefing and email.

The combined briefing overview is also retried once. If both attempts fail, the
email and saved briefing state the overview error and still include every valid
individual article summary.

### Long articles

Set `CHUNKED_SUMMARIZATION=true` to process the entire article without the
`MAX_ARTICLE_CHARACTERS` truncation. Articles longer than
`SUMMARY_CHUNK_CHARACTERS` are split into large chunks, each chunk is
summarized independently, and the chunk summaries are combined into the normal
article summary. An article of N chunks normally needs N+1 summary requests;
extra combining requests are used if the chunk summaries themselves exceed the
chunk limit. Shorter articles use one summary request. Relevance filtering still
uses only its configured excerpt.

The default chunk limit is 9,500 characters, chosen for a 4,096-token context.
Characters are not tokens, and prompts and model responses also consume
context. Adjust `SUMMARY_CHUNK_CHARACTERS` for the allocated context and text;
even the default does not guarantee that a request fits. Increase it for a
larger context to reduce calls, or lower it for unusually token-dense text. Long articles
require more calls and intermediate summaries can omit details. If a chunk or
combine request fails, the whole article is retried once and remains in the
landing folder if it still fails.

Set `CHUNKED_SUMMARIZATION=false` to restore the single-request path and its
`MAX_ARTICLE_CHARACTERS` cap. If the variable is omitted, chunking is disabled.

## AI relevance filtering

Before creating an article summary, Ollama evaluates each landing-file article
against the editorial criteria in
[`config/relevance_criteria.txt`](config/relevance_criteria.txt). Edit that file
to control what is worthy of the email.

The relevance response must be JSON with an `include` boolean and a non-empty
`reason`. A valid `include: false` archives the source article without
summarizing it. The email and saved briefing list its title, original link (if
available), and the model's reason under “Articles not selected.” Summary
failures and malformed landing files also appear there with the available
details and failure reason. Invalid relevance output is retried up to three
times. If all three attempts are invalid, the article is summarized and placed
in a clearly marked section at the bottom of the email and saved briefing,
outside the combined briefing overview. That section includes all three invalid
model outputs for review.

The top of every email and saved briefing reports the number of articles
included in the main briefing and not included, including malformed files and
relevance-triage failures. The email can still list those not included.

Malformed landing files do not stop valid files from being processed. The
command prints a warning, includes the warning in `briefing.json`, and leaves
each malformed file in `data/landing/` for correction. Successfully summarized
and relevance-rejected files are archived after successful delivery (or after
saving if email is disabled). Errors from Ollama or briefing-output creation
can still stop the batch before archiving.

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | Yes | Model passed to Ollama for every generation |
| `OLLAMA_BASE_URL` | No | Defaults to `http://127.0.0.1:11434` |
| `OLLAMA_TIMEOUT_SECONDS` | No | Request timeout, default `180` |
| `LANDING_DIR` | No | Input directory, default `data/landing` |
| `ARCHIVE_DIR` | No | Archived input directory, default `data/archive` |
| `BRIEFING_DIR` | No | Generated briefing directory, default `data/briefings` |
| `MAX_ARTICLE_CHARACTERS` | No | Per-article input cap when chunking is disabled, default `24000` |
| `CHUNKED_SUMMARIZATION` | No | Enable full-article chunking, default `false`; `.env.example` enables it |
| `SUMMARY_CHUNK_CHARACTERS` | No | Approximate per-chunk character limit, default `9500` |
| `RELEVANCE_CRITERIA_FILE` | No | Criteria text file for AI relevance filtering |
| `RELEVANCE_MAX_ARTICLE_CHARACTERS` | No | Article excerpt used for relevance triage, default `6000` |
| `SMTP_ENABLED` | No | Set to `true` to email a successfully generated briefing |
| `SMTP_HOST` | When enabled | SMTP hostname, for example your mail provider's SMTP server |
| `SMTP_PORT` | When enabled | SMTP port, default `587` |
| `SMTP_USERNAME` | When enabled | SMTP login username |
| `SMTP_PASSWORD` | When enabled | SMTP password or provider app password |
| `SMTP_FROM` | When enabled | Sender email address |
| `SMTP_TO` | When enabled | Comma-separated recipient email addresses |
| `SMTP_USE_STARTTLS` | No | Use STARTTLS for SMTP, default `true` |
| `SMTP_USE_SSL` | No | Use implicit SSL/TLS, default `false`; do not enable with STARTTLS |
| `SMTP_TIMEOUT_SECONDS` | No | SMTP connection timeout, default `30` |

Do not commit `.env`; it may later contain SMTP credentials.
`.env.example` contains only safe placeholders and should be committed so people
cloning the project can create their own local configuration.

## SMTP email delivery

Email is disabled by default. To enable it, copy the SMTP values from your mail
provider into your ignored `.env` file and set:

```dotenv
SMTP_ENABLED=true
```

Use an app password when your provider requires one; do not use or commit a
personal account password. When SMTP is enabled, the generated briefing email
contains the short overview followed by the individual article summaries and
their timings. Source files move from `data/landing/` to `data/archive/` only
after SMTP delivery succeeds. If delivery fails, the files remain in the
landing folder for a later retry.

If a run aborts, an enabled SMTP sender attempts a separate failure email with
the error, elapsed time, the number of articles fully processed (including
rejections and handled per-article failures), and the file being processed, if
any. A failure outside article processing is reported as such. If notification
delivery also fails, the original error and the notification error are reported
on stderr. No failure email can be sent if SMTP is disabled or its configuration
cannot be loaded. Handled article and overview errors remain in the normal
briefing rather than triggering a separate failure email.