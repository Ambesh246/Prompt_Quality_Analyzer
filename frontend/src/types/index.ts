export type Analysis = {
  score: number;
  category: string;
  assessment: string;
  rubric_score: number;
  model_score: number | null;
  model_preference_score: number | null;
  scoring_method: string;
  dimensions: Record<string, number>;
  dimension_labels: Record<string, string>;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  features: Record<string, number | string | boolean>;
  model: {
    available: boolean;
    version: string;
    training_data: string;
    label_semantics: string;
    metrics: Record<string, number>;
    limitations: string[];
    load_error?: string | null;
    scoring_mode?: string;
    artifact_filename?: string | null;
  };
};

export type Improvement = {
  status: string;
  provider?: string | null;
  model?: string | null;
  original_prompt: string;
  rewritten_prompt?: string | null;
  original_analysis: Analysis;
  improved_analysis?: Analysis | null;
  score_delta?: number | null;
  changes: string[];
  notice?: string | null;
  error?: string | null;
};
