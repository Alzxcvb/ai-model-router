/**
 * Prompt classifier — keyword rules + LLM-based classification.
 * Mirrors router/rules.py and router/llm_classifier.py.
 */

import type { TaskType } from "./models";

export interface ClassificationResult {
  taskType: TaskType;
  confidence: number;
  complexity: "low" | "medium" | "high";
  needsReasoning: boolean;
  needsCreativity: boolean;
  keywordsMatched: string[];
  method: "rules" | "llm" | "llm_fallback";
}

const RULES: Record<string, string[]> = {
  code: [
    "write a function", "write a script", "implement", "debug", "fix this bug",
    "refactor", "code review", "unit test", "regex", "algorithm",
    "python", "javascript", "typescript", "java", "rust", "golang",
    "html", "css", "sql", "api endpoint", "class", "function",
    "compile", "runtime error", "syntax error", "stack trace",
  ],
  writing: [
    "essay", "blog post", "article", "creative writing", "story",
    "poem", "rewrite", "proofread", "persuasive", "narrative",
    "write me", "draft", "compose", "copywriting", "slogan",
  ],
  reasoning: [
    "solve", "calculate", "prove", "logic", "math", "equation",
    "why does", "explain why", "what would happen if", "probability",
    "derive", "reasoning", "step by step", "analyze the argument",
  ],
  summarization: [
    "summarize", "tldr", "tl;dr", "key points", "condense",
    "brief overview", "main ideas", "recap", "in short",
  ],
  conversation: [
    "chat", "tell me about yourself", "how are you", "what do you think",
    "let's talk", "opinion on", "recommend",
  ],
  research: [
    "research", "find information", "what is", "who is", "when did",
    "compare and contrast", "pros and cons", "sources", "evidence",
    "literature review", "state of the art",
  ],
  translation: [
    "translate", "translation", "in spanish", "in french", "in german",
    "in japanese", "in chinese", "in arabic", "to english", "from english",
    "localize", "multilingual",
  ],
  data: [
    "csv", "json", "parse", "data analysis", "spreadsheet",
    "table", "dataset", "extract data", "structured data",
    "visualization", "chart", "graph", "statistics",
  ],
};

export function classifyByRules(prompt: string): ClassificationResult {
  const lower = prompt.toLowerCase();
  const scores: Record<string, string[]> = {};

  for (const [taskType, keywords] of Object.entries(RULES)) {
    scores[taskType] = [];
    for (const kw of keywords) {
      const regex = new RegExp(`\\b${kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
      if (regex.test(lower)) {
        scores[taskType].push(kw);
      }
    }
  }

  let bestType = "conversation";
  let bestCount = 0;
  for (const [taskType, matched] of Object.entries(scores)) {
    if (matched.length > bestCount) {
      bestCount = matched.length;
      bestType = taskType;
    }
  }

  if (bestCount === 0) {
    return {
      taskType: "conversation" as TaskType,
      confidence: 0.3,
      complexity: "low",
      needsReasoning: false,
      needsCreativity: false,
      keywordsMatched: [],
      method: "rules",
    };
  }

  const totalKeywords = RULES[bestType]?.length ?? 1;
  const confidence = Math.min(0.9, 0.4 + (bestCount / totalKeywords) * 0.5);

  return {
    taskType: bestType as TaskType,
    confidence: Math.round(confidence * 100) / 100,
    complexity: "medium",
    needsReasoning: bestType === "reasoning",
    needsCreativity: bestType === "writing",
    keywordsMatched: scores[bestType] ?? [],
    method: "rules",
  };
}
