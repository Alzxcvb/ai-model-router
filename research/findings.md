# Research Findings — AI Model Router

Running notes on observations, discoveries, and insights gathered during development.

## 1. Classification Accuracy

### Rules Engine (v0.1)
- Keyword matching works surprisingly well for **clear-intent prompts** (e.g., "Write a Python function" → code, "Translate to Spanish" → translation).
- Falls apart on **ambiguous prompts** that span categories:
  - "Explain the quicksort algorithm step by step" — is this `code`, `reasoning`, or `research`? The rules engine picks `code` (matches "algorithm"), but the user wants an explanation, not code.
  - "What are the pros and cons of TypeScript?" — matches both `research` ("pros and cons") and `code` ("TypeScript"). Rules pick `code` due to more keyword matches, but this is really a research/opinion question.
- **Confidence calibration** is crude — based solely on keyword count ratio, not actual certainty. A prompt with 3 code keywords gets high confidence even if one writing keyword is a better signal.

### LLM Classifier (v0.2)
- Gemini Flash classifies ambiguous prompts much more accurately because it understands intent, not just keywords.
- Adds ~200-400ms latency and ~$0.00005 cost per classification — negligible.
- The `complexity` field is valuable: allows budget downgrading for simple prompts (e.g., "What is 2+2?" doesn't need Claude Sonnet).
- `needs_reasoning` and `needs_creativity` flags open the door for more nuanced model selection beyond task type alone.

### Key Insight
> A two-stage approach (cheap LLM classifier → expensive task model) saves money overall. The classifier cost is 100-1000x cheaper than the task model, and better routing avoids wasting expensive models on simple tasks.

## 2. Model Strengths and Weaknesses

Observations based on published benchmarks, community testing, and our own prompt testing:

| Model | Strengths | Weaknesses | Sweet Spot |
|-------|-----------|------------|------------|
| **Claude Sonnet 4.5** | Writing quality, nuance, instruction following, long-form output | Higher cost, sometimes verbose | Writing, complex reasoning, creative tasks |
| **GPT-4o** | Well-rounded, strong tool use, good at structured output | Occasionally formulaic writing | Data analysis, translation, balanced tasks |
| **Gemini 2.0 Flash** | Extremely cheap, fast, huge context window (1M tokens) | Lower quality on complex tasks | Summarization, simple Q&A, classification |
| **DeepSeek V3** | Excellent code generation, strong math/reasoning | Weak creative writing, less nuanced | Code, data analysis, algorithmic problems |
| **Llama 3.3 70B** | Open source, good general performance, low cost | Not best-in-class at anything | Budget-conscious general use |
| **Qwen 2.5 72B** | Strong multilingual, good code, competitive pricing | Less tested in English-centric tasks | Translation, multilingual tasks, code |

### Surprising Findings
1. **DeepSeek V3 matches Claude/GPT on code** at ~10x lower cost. For pure code generation, it's arguably the best value.
2. **Gemini Flash is underrated for summarization** — its huge context window and low cost make it ideal for condensing long documents.
3. **No single model wins everything** — this validates the entire premise of the router.
4. **Writing quality is the most subjective category** — model rankings here vary most by evaluator preference.

## 3. Cost/Performance Tradeoffs

### Cost per 1M tokens (input/output)
```
Model                    Input     Output    Total
─────────────────────────────────────────────────
Gemini 2.0 Flash         $0.10     $0.40     $0.50
DeepSeek V3              $0.27     $1.10     $1.37
Qwen 2.5 72B             $0.35     $0.40     $0.75
Llama 3.3 70B            $0.40     $0.40     $0.80
GPT-4o                   $2.50    $10.00    $12.50
Claude Sonnet 4.5        $3.00    $15.00    $18.00
```

### Score-to-Cost Ratio (higher = better value)
For code tasks:
- DeepSeek V3: 9.0 / $1.37 = **6.57** (best value)
- Qwen 2.5: 8.0 / $0.75 = 10.67
- Gemini Flash: 7.0 / $0.50 = **14.0** (if quality is sufficient)
- Claude: 9.0 / $18.00 = 0.50 (worst value, but highest quality)

### Key Insight
> The "balanced" budget mode captures most of the quality at a fraction of the cost. For many tasks, the difference between a 9/10 and 7/10 model is barely noticeable, but the cost difference is 20-35x.

## 4. Router Value Proposition

### When the router adds value:
1. **Mixed workloads** — an app handling code, writing, and Q&A benefits from routing each to the best model.
2. **Cost optimization** — routing simple tasks to cheap models saves 90%+ vs. always using the best model.
3. **Latency optimization** — Flash models respond 2-5x faster for simple tasks.
4. **Transparency** — showing users *why* a model was chosen builds trust and enables debugging.

### When just using one model is fine:
1. **Single task type** — if your app only does code generation, just use the best code model.
2. **Low volume** — the cost savings don't matter if you're making 10 requests/day.
3. **Quality is paramount** — if you always need the absolute best, routing adds complexity without benefit.

## 5. Limitations Discovered

1. **Benchmark scores are static** — model capabilities change with updates, but our registry is manually maintained.
2. **Task type is too coarse** — "code" includes everything from "fix this typo" to "implement a compiler". Complexity matters more than we initially modeled.
3. **Evaluation is subjective** — LLM-as-judge scoring varies by judge model and prompt. Our evaluator using Claude may be biased toward Claude-style responses.
4. **No feedback loop yet** — the router can't learn from user satisfaction. A user clicking "regenerate" is a signal we're not capturing.
5. **Context window usage isn't considered** — a long document summarization should prefer Gemini Flash (1M context) over models with 128K limits.

## 6. Future Research Directions

1. **Adaptive scoring** — update model scores based on actual benchmark results, not just published numbers.
2. **Multi-model consensus** — for high-stakes tasks, route to 2-3 models and pick the best response.
3. **Prompt rewriting** — optimize the prompt for the selected model's strengths.
4. **User preference learning** — track which responses users prefer and adjust routing weights.
5. **Latency-aware routing** — consider current model load/latency, not just benchmark scores.
6. **Domain-specific routing** — "code" in Python vs. Rust vs. Haskell may have different optimal models.
