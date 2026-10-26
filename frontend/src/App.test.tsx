import { describe, expect, it, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';
import { analyzePrompt, improvePrompt } from './services/api';
import type { Analysis, Improvement } from './types';

vi.mock('./services/api', () => ({
  analyzePrompt: vi.fn(),
  improvePrompt: vi.fn(),
}));

const sampleAnalysis: Analysis = {
  score: 72,
  category: 'Good',
  assessment: 'The prompt gives a useful direction.',
  rubric_score: 76,
  model_score: 56,
  model_preference_score: 56,
  scoring_method: '80%_rubric_plus_20%_learned_pair_preference_proxy',
  dimensions: {
    clarity: 80, context: 65, specificity: 71, task_definition: 85, constraints: 55,
    output_format: 60, audience_purpose: 50, completeness: 72, actionability: 80,
    ambiguity_redundancy: 87,
  },
  dimension_labels: {
    clarity: 'Clarity', context: 'Context', specificity: 'Specificity', task_definition: 'Task definition',
    constraints: 'Constraints', output_format: 'Output format', audience_purpose: 'Audience & purpose',
    completeness: 'Completeness', actionability: 'Actionability', ambiguity_redundancy: 'Ambiguity & redundancy',
  },
  strengths: ['Clarity is a relative strength (80/100).'],
  weaknesses: ['Constraints needs attention (55/100).'],
  recommendations: ['State relevant limits or requirements.'],
  features: { word_count: 5, character_count: 20 },
  model: {
    available: true, version: 'pqa-seed-synthetic-v0.1',
    training_data: 'synthetic seed prompts', label_semantics: 'relative preference', metrics: { holdout_f1: 1 }, limitations: [],
  },
};

const sampleImprovement: Improvement = {
  status: 'success', provider: 'rule_based_fallback', model: 'deterministic-v1',
  original_prompt: 'Explain photosynthesis',
  rewritten_prompt: 'Explain photosynthesis in simple terms, with one brief example.',
  original_analysis: sampleAnalysis,
  improved_analysis: { ...sampleAnalysis, score: 80, category: 'Excellent' },
  score_delta: 8,
  changes: ['Made the preferred level of explanation and example explicit.'],
  notice: 'This is a deterministic template-based rewrite, not an LLM-generated rewrite.',
};

describe('Prompt Quality Analyzer UI', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(analyzePrompt).mockResolvedValue(sampleAnalysis);
    vi.mocked(improvePrompt).mockResolvedValue(sampleImprovement);
  });

  it('prevents submitting a blank prompt and shows validation feedback', async () => {
    render(<App />);
    const analyzeButton = screen.getByRole('button', { name: /analyze prompt/i });
    expect(analyzeButton).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/enter the prompt/i), { target: { value: '   ' } });
    expect(analyzeButton).toBeDisabled();
    expect(analyzePrompt).not.toHaveBeenCalled();
  });

  it('submits a prompt and renders score plus all dimension results', async () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText(/enter the prompt/i), { target: { value: 'Explain photosynthesis' } });
    fireEvent.click(screen.getByRole('button', { name: /analyze prompt/i }));
    expect(await screen.findByText('At a glance')).toBeInTheDocument();
    expect(screen.getByLabelText('Prompt score 72 out of 100')).toBeInTheDocument();
    expect(screen.getByText('Quality breakdown')).toBeInTheDocument();
    expect(screen.getByText('Ambiguity & redundancy')).toBeInTheDocument();
    expect(analyzePrompt).toHaveBeenCalledWith('Explain photosynthesis');
  });

  it('improves a prompt with the no-provider fallback and renders a before/after comparison', async () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText(/enter the prompt/i), { target: { value: 'Explain photosynthesis' } });
    fireEvent.click(screen.getByRole('button', { name: /analyze prompt/i }));
    fireEvent.click(await screen.findByRole('button', { name: /improve prompt/i }));
    expect(await screen.findByText('Compare before and after')).toBeInTheDocument();
    expect(screen.getByDisplayValue(sampleImprovement.rewritten_prompt!)).toBeInTheDocument();
    expect(improvePrompt).toHaveBeenCalledWith('Explain photosynthesis', false);
  });

  it('shows a useful error if analysis API fails', async () => {
    vi.mocked(analyzePrompt).mockRejectedValue(new Error('Backend unavailable'));
    render(<App />);
    fireEvent.change(screen.getByLabelText(/enter the prompt/i), { target: { value: 'Explain photosynthesis' } });
    fireEvent.click(screen.getByRole('button', { name: /analyze prompt/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Backend unavailable');
  });
});

describe("Loading and clipboard interactions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(analyzePrompt).mockResolvedValue(sampleAnalysis);
    vi.mocked(improvePrompt).mockResolvedValue(sampleImprovement);
  });
it('shows the loading label while analysis is pending', async () => {
  vi.mocked(analyzePrompt).mockImplementation(() => new Promise<Analysis>(() => {}));
  render(<App />);
  fireEvent.change(screen.getByLabelText(/enter the prompt/i), { target: { value: 'Explain photosynthesis' } });
  fireEvent.click(screen.getByRole('button', { name: /analyze prompt/i }));
  expect(await screen.findByRole('button', { name: /analyzing prompt/i })).toBeDisabled();
});

it('copies the editable improved prompt and confirms after clipboard success', async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText },
  });
  render(<App />);
  fireEvent.change(screen.getByLabelText(/enter the prompt/i), { target: { value: 'Explain photosynthesis' } });
  fireEvent.click(screen.getByRole('button', { name: /analyze prompt/i }));
  fireEvent.click(await screen.findByRole('button', { name: /improve prompt/i }));
  const copyButton = await screen.findByRole('button', { name: /copy prompt/i });
  fireEvent.click(copyButton);
  await waitFor(() => expect(writeText).toHaveBeenCalledWith(sampleImprovement.rewritten_prompt));
  expect(await screen.findByRole('button', { name: /copied/i })).toBeInTheDocument();
});
});
