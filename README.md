# AI Model Router

Intelligent multi-model orchestration — routes AI prompts to the best model for each task.

Different AI models excel at different tasks. Claude leads in writing, DeepSeek matches it on code at 10x lower cost, Gemini Flash handles simple tasks for nearly free. This project classifies prompts by task type and routes them to the optimal model based on quality, cost, and complexity.

**[Read the full research write-up](research/writeup.md)**

## How It Works

```
User prompt: "Write a Python quicksort"
    |
    v
Classifier -----> task_type: code, confidence: 0.92
    |
    v
Router ----------> Claude Sonnet 4.5 (code score: 9/10)
    |              DeepSeek V3 also 9/10 but cheaper
    |              Gemini Flash 7/10 but 36x cheaper
    v
Provider --------> OpenRouter API call
    |
    v
Response + metadata (model, latency, cost, reasoning)
```

## Quick Start

```bash
git clone https://github.com/Alzxcvb/ai-model-router.git
cd ai-model-router

# Python setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=your-key" > .env
```

### CLI Usage

```bash
# Dry run — see routing decision without calling any API
python -m router --dry-run "Write a Python quicksort"
python -m router --dry-run "Write a persuasive essay on climate change"
python -m router --dry-run --budget cheap "What is 15% of 847?"

# Live — actually calls the selected model
python -m router "Write a Python quicksort"
python -m router --budget balanced "Summarize this article..."
python -m router --classifier llm "Explain quantum computing"
```

**Flags:**
- `--budget` — `best` (highest quality), `balanced` (score/cost ratio), `cheap` (lowest cost with score >= 7)
- `--classifier` — `rules` (keyword matching, free) or `llm` (Gemini Flash, ~$0.00005/call)
- `--dry-run` — show routing decision without calling the model

### Web Demo

```bash
cd web
npm install
npm run build
npm start
# Open http://localhost:3000
```

Chat interface with a side panel showing the full routing pipeline: classification, model selection reasoning, score comparisons, and cost estimates.

### Benchmarks

```bash
# Preview what would run
python -m benchmarks.runner --dry-run

# Run benchmarks (top 2 models per task type)
python -m benchmarks.runner --models 2

# Score responses with LLM judge
python -m benchmarks.evaluator benchmarks/data/results/run_XXXXXXXX_XXXXXX.json
```

## Architecture

```
ai-model-router/
├── router/                    # Python routing engine
│   ├── types.py               #   Data classes (TaskType, ModelInfo, RoutingDecision)
│   ├── rules.py               #   Keyword-based classifier (v0.1)
│   ├── llm_classifier.py      #   LLM-based classifier via Gemini Flash (v0.2)
│   ├── classifier.py          #   Classifier facade (rules or llm)
│   ├── models.py              #   Model registry — 6 models, 8 task types, scores + costs
│   ├── router.py              #   Core: classify → select → call
│   ├── providers.py           #   OpenRouter API wrapper
│   └── __main__.py            #   CLI entry point
├── benchmarks/
│   ├── data/prompts.json      #   19 curated prompts with eval criteria
│   ├── runner.py              #   Sends prompts to multiple models
│   └── evaluator.py           #   LLM-judge scoring (Claude Sonnet)
├── web/                       # TypeScript web demo
│   ├── src/server.ts          #   Express API server
│   ├── src/models.ts          #   Model registry (TypeScript mirror)
│   ├── src/classifier.ts      #   Rules classifier (TypeScript mirror)
│   └── public/                #   Chat UI (HTML/CSS/JS)
├── research/
│   ├── writeup.md             #   Full research paper
│   └── findings.md            #   Running research notes
└── tests/                     # 37 tests (pytest)
```

## Models Supported

| Model | Best At | Cost (per 1M tokens) |
|-------|---------|---------------------|
| Claude Sonnet 4.5 | Writing, reasoning | $3.00 in / $15.00 out |
| GPT-4o | Translation, data analysis | $2.50 in / $10.00 out |
| DeepSeek V3 | Code, math | $0.27 in / $1.10 out |
| Qwen 2.5 72B | Multilingual, code | $0.35 in / $0.40 out |
| Llama 3.3 70B | General purpose | $0.40 in / $0.40 out |
| Gemini 2.0 Flash | Summarization, simple tasks | $0.10 in / $0.40 out |

## Task Categories

`code` | `writing` | `reasoning` | `summarization` | `conversation` | `research` | `translation` | `data`

## Key Findings

- **No single model wins everything** — validates the routing approach
- **Balanced routing saves ~91%** vs. always using Claude, with minimal quality impact
- **DeepSeek V3 matches Claude/GPT on code** at ~10x lower cost
- **LLM classification adds ~$0.00005/call** — negligible cost for much better accuracy
- **Keyword rules achieve 89% accuracy** on clear-intent prompts

See [research/writeup.md](research/writeup.md) for the full analysis.

## Tests

```bash
python -m pytest tests/ -v   # 37 tests
```

## License

MIT
