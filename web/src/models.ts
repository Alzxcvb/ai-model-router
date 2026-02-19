/**
 * Model registry — mirrors the Python router/models.py.
 * Scores, costs, and metadata for each supported model.
 */

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  scores: Record<string, number>;
  costPerMillionInput: number;
  costPerMillionOutput: number;
  maxContext: number;
  supportsImages: boolean;
}

export const TASK_TYPES = [
  "code",
  "writing",
  "reasoning",
  "summarization",
  "conversation",
  "research",
  "translation",
  "data",
] as const;

export type TaskType = (typeof TASK_TYPES)[number];

export const MODELS: Record<string, ModelInfo> = {
  "anthropic/claude-sonnet-4-5": {
    id: "anthropic/claude-sonnet-4-5",
    name: "Claude Sonnet 4.5",
    provider: "anthropic",
    scores: {
      code: 9, writing: 10, reasoning: 9, summarization: 9,
      conversation: 9, research: 9, translation: 8, data: 8.5,
    },
    costPerMillionInput: 3.0,
    costPerMillionOutput: 15.0,
    maxContext: 200_000,
    supportsImages: true,
  },
  "openai/gpt-4o": {
    id: "openai/gpt-4o",
    name: "GPT-4o",
    provider: "openai",
    scores: {
      code: 9, writing: 8, reasoning: 9, summarization: 8,
      conversation: 8.5, research: 8.5, translation: 9, data: 9,
    },
    costPerMillionInput: 2.5,
    costPerMillionOutput: 10.0,
    maxContext: 128_000,
    supportsImages: true,
  },
  "google/gemini-2.0-flash-001": {
    id: "google/gemini-2.0-flash-001",
    name: "Gemini 2.0 Flash",
    provider: "google",
    scores: {
      code: 7, writing: 7, reasoning: 7, summarization: 8,
      conversation: 7.5, research: 7.5, translation: 7.5, data: 7.5,
    },
    costPerMillionInput: 0.1,
    costPerMillionOutput: 0.4,
    maxContext: 1_000_000,
    supportsImages: true,
  },
  "deepseek/deepseek-chat-v3": {
    id: "deepseek/deepseek-chat-v3",
    name: "DeepSeek V3",
    provider: "deepseek",
    scores: {
      code: 9, writing: 6, reasoning: 8.5, summarization: 7,
      conversation: 6.5, research: 7, translation: 7, data: 8.5,
    },
    costPerMillionInput: 0.27,
    costPerMillionOutput: 1.10,
    maxContext: 128_000,
    supportsImages: false,
  },
  "meta-llama/llama-3.3-70b-instruct": {
    id: "meta-llama/llama-3.3-70b-instruct",
    name: "Llama 3.3 70B",
    provider: "meta",
    scores: {
      code: 7.5, writing: 7, reasoning: 7.5, summarization: 7.5,
      conversation: 7, research: 7, translation: 7, data: 7,
    },
    costPerMillionInput: 0.40,
    costPerMillionOutput: 0.40,
    maxContext: 128_000,
    supportsImages: false,
  },
  "qwen/qwen-2.5-72b-instruct": {
    id: "qwen/qwen-2.5-72b-instruct",
    name: "Qwen 2.5 72B",
    provider: "qwen",
    scores: {
      code: 8, writing: 7, reasoning: 8, summarization: 7.5,
      conversation: 7, research: 7, translation: 8.5, data: 8,
    },
    costPerMillionInput: 0.35,
    costPerMillionOutput: 0.40,
    maxContext: 128_000,
    supportsImages: false,
  },
};

export function getBestModelForTask(taskType: TaskType, budget: "best" | "balanced" | "cheap" = "best"): ModelInfo {
  const models = Object.values(MODELS);

  if (budget === "cheap") {
    const candidates = models.filter((m) => (m.scores[taskType] ?? 0) >= 7);
    if (candidates.length > 0) {
      return candidates.reduce((a, b) => a.costPerMillionInput < b.costPerMillionInput ? a : b);
    }
  }

  if (budget === "balanced") {
    return models.reduce((a, b) => {
      const ratioA = (a.scores[taskType] ?? 0) / Math.max(a.costPerMillionInput + a.costPerMillionOutput, 0.01);
      const ratioB = (b.scores[taskType] ?? 0) / Math.max(b.costPerMillionInput + b.costPerMillionOutput, 0.01);
      return ratioA >= ratioB ? a : b;
    });
  }

  // "best"
  return models.reduce((a, b) => (a.scores[taskType] ?? 0) >= (b.scores[taskType] ?? 0) ? a : b);
}

export function getRankedModels(taskType: TaskType): Array<{ model: ModelInfo; score: number }> {
  return Object.values(MODELS)
    .map((model) => ({ model, score: model.scores[taskType] ?? 0 }))
    .sort((a, b) => b.score - a.score);
}
