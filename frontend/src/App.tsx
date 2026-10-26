import { useMemo, useState, type CSSProperties } from 'react';
import {
  ArrowDown, ArrowRight, ArrowUpRight, BarChart3, Check, CheckCircle2, ChevronDown,
  CircleHelp, Code2, Copy, Download, FileText, FlaskConical, Lightbulb,  LoaderCircle, Moon, RotateCcw, ShieldCheck, Sparkles, Sun, Target, WandSparkles,
  X, Zap,
} from 'lucide-react';
import { analyzePrompt, improvePrompt } from './services/api';
import type { Analysis, Improvement } from './types';

const EXAMPLES = [
  {
    label: 'Academic', icon: FileText,
    prompt: 'Explain climate change.',
  },
  {
    label: 'Coding', icon: Code2,
    prompt: 'Help me fix this Python function that returns duplicate values. Explain the bug, show corrected code, and add two pytest tests.',
  },
  {
    label: 'Data analysis', icon: BarChart3,
    prompt: 'Analyze my sales dataset.',
  },
];

const FRIENDLY_DIMENSIONS: Record<string, string> = {
  clarity: 'Clarity', context: 'Context', specificity: 'Specificity',
  task_definition: 'Task definition', constraints: 'Constraints',
  output_format: 'Output format', audience_purpose: 'Audience & purpose',
  completeness: 'Completeness', actionability: 'Actionability',
  ambiguity_redundancy: 'Ambiguity & redundancy',
};

function scoreColor(score: number) {
  if (score >= 80) return 'excellent';
  if (score >= 60) return 'good';
  if (score >= 40) return 'fair';
  return 'needs-work';
}

function shortModelName(version: string) {
  if (version.includes('ridge')) return 'Colab Ridge';
  if (version.includes('synthetic')) return 'Synthetic seed';
  if (version.includes('bpo')) return 'BPO preference';
  return version;
}

function ScoreRing({ score, large = false }: { score: number; large?: boolean }) {
  const cls = scoreColor(score);
  return (
    <div className={`score-ring ${large ? 'score-ring-large' : ''} ${cls}`} style={{ '--score': score } as CSSProperties} aria-label={`Prompt score ${Math.round(score)} out of 100`}>
      <div className="score-ring-center">
        <span className="score-value">{Math.round(score)}</span>
        <span className="score-outof">/ 100</span>
      </div>
    </div>
  );
}

function ScorePill({ score }: { score: number }) {
  return <span className={`score-pill ${scoreColor(score)}`}>{Math.round(score)} / 100</span>;
}

function DimensionBars({ analysis }: { analysis: Analysis }) {
  return (
    <div className="dimension-list">
      {Object.entries(analysis.dimensions).map(([key, value]) => (
        <div className="dimension-row" key={key}>
          <div className="dimension-meta">
            <span>{analysis.dimension_labels[key] || FRIENDLY_DIMENSIONS[key] || key}</span>
            <strong>{Math.round(value)}</strong>
          </div>
          <div className="dimension-track" role="progressbar" aria-label={analysis.dimension_labels[key] || key} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value)}>
            <div className={`dimension-fill ${scoreColor(value)}`} style={{ width: `${value}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [improvement, setImprovement] = useState<Improvement | null>(null);
  const [loadingAnalyze, setLoadingAnalyze] = useState(false);
  const [loadingImprove, setLoadingImprove] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [useProvider, setUseProvider] = useState(false);
  const [showMethod, setShowMethod] = useState(true);
  const [editableRewrite, setEditableRewrite] = useState<string | null>(null);
  const [rewriteEditDirty, setRewriteEditDirty] = useState(false);
  const [recheckingRewrite, setRecheckingRewrite] = useState(false);

  const wordCount = useMemo(() => prompt.trim() ? prompt.trim().split(/\s+/).length : 0, [prompt]);
  const rewrittenText = editableRewrite ?? improvement?.rewritten_prompt ?? '';

  function resetResults() {
    setAnalysis(null);
    setImprovement(null);
    setError('');
    setCopied(false);
    setEditableRewrite(null);
    setRewriteEditDirty(false);
    setRecheckingRewrite(false);
    setShowMethod(false);
  }

  async function handleAnalyze() {
    if (!prompt.trim()) {
      setError('Add a prompt first so there is something to analyze.');
      return;
    }
    setLoadingAnalyze(true);
    setError('');
    setAnalysis(null);
    setImprovement(null);
    setEditableRewrite(null);
    setRewriteEditDirty(false);
    setCopied(false);
    try {
      const result = await analyzePrompt(prompt.trim());
      setAnalysis(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
    } finally {
      setLoadingAnalyze(false);
    }
  }

  async function handleImprove(providerOverride = useProvider) {
    if (!prompt.trim() || !analysis) return;
    setLoadingImprove(true);
    setError('');
    setCopied(false);
    try {
      const result = await improvePrompt(prompt.trim(), providerOverride);
      setImprovement(result);
      if (result.status === 'success' && result.rewritten_prompt) {
        setEditableRewrite(result.rewritten_prompt);
        setRewriteEditDirty(false);
      } else if (result.error) {
        setError(result.error);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The rewrite request failed. Your original analysis is still available.');
    } finally {
      setLoadingImprove(false);
    }
  }

  async function rescoreEditedRewrite() {
    if (!rewrittenText.trim() || !improvement) {
      setError('Add some text to the improved prompt before re-scoring it.');
      return;
    }
    setRecheckingRewrite(true);
    setError('');
    try {
      const updatedAnalysis = await analyzePrompt(rewrittenText.trim());
      setImprovement((current) => {
        if (!current) return current;
        return {
          ...current,
          improved_analysis: updatedAnalysis,
          score_delta: Math.round((updatedAnalysis.score - current.original_analysis.score) * 10) / 10,
        };
      });
      setRewriteEditDirty(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not score the edited prompt.');
    } finally {
      setRecheckingRewrite(false);
    }
  }

  async function copyRewrite() {
    if (!rewrittenText) return;
    try {
      await navigator.clipboard.writeText(rewrittenText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2200);
    } catch {
      setError('Clipboard access was blocked by the browser. Select the rewritten text and copy it manually.');
    }
  }

  function downloadRewrite() {
    if (!rewrittenText) return;
    const blob = new Blob([rewrittenText], { type: 'text/plain;charset=utf-8' });
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = 'improved-prompt.txt';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1500);
  }

  function loadExample(examplePrompt: string) {
    setPrompt(examplePrompt);
    resetResults();
  }

  return (
    <div className={`app-shell ${darkMode ? 'dark-mode' : ''}`}>
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Promptly home">
          <span className="brand-mark"><Sparkles size={19} strokeWidth={2.4} /></span>
          <span>promptly<span className="brand-dot">.</span></span>
        </a>
        <nav className="top-nav" aria-label="Main navigation">
          <a href="#analyzer">Analyzer</a>
          <a href="#how-it-works">How it works</a>
        </nav>
        <div className="top-actions">
          <span className="privacy-chip"><ShieldCheck size={14} /> Privacy-first</span>
          <button className="icon-button theme-toggle" onClick={() => setDarkMode(!darkMode)} aria-label={darkMode ? 'Switch to light theme' : 'Switch to dark theme'} title={darkMode ? 'Switch to light theme' : 'Switch to dark theme'}>
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </header>

      <main id="top" className="main-content">
        <section className="hero-section">
          <div className="eyebrow"><span className="eyebrow-pulse" /> YOUR PROMPTS, UPGRADED</div>
          <h1>Write better prompts.<br /><span>Get clearer AI results.</span></h1>
          <p className="hero-copy">A thoughtful prompt makes a difference. See what is working, spot what is missing, and improve your instructions before you send them to any AI.</p>
          <div className="hero-proof-row">
            <span><CheckCircle2 size={16} /> 10 quality dimensions</span>
            <span><CheckCircle2 size={16} /> Actionable feedback</span>
            <span><CheckCircle2 size={16} /> Works across LLMs</span>
          </div>
        </section>

        <section id="analyzer" className="workspace-grid">
          <div className="composer-card surface-card">
            <div className="card-heading-row">
              <div>
                <div className="section-kicker">PROMPT WORKSPACE</div>
                <h2>What do you want to ask?</h2>
              </div>
              <span className="input-count">{prompt.length.toLocaleString()} / 8,000</span>
            </div>
            <label htmlFor="prompt-input" className="sr-only">Enter the prompt you want to analyze</label>
            <textarea
              id="prompt-input"
              className="prompt-textarea"
              placeholder="Paste your prompt here…\n\nFor example: Explain photosynthesis to a beginner in simple terms, then give one example."
              value={prompt}
              maxLength={8000}
              disabled={loadingAnalyze || loadingImprove}
              onChange={(event) => {
                setPrompt(event.target.value);
                if (analysis || improvement) resetResults();
                else if (error) setError('');
              }}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') void handleAnalyze();
              }}
            />
            <div className="composer-bottom-line">
              <span className="word-counter">{wordCount} {wordCount === 1 ? 'word' : 'words'}</span>
              <div className="composer-bottom-actions">
                <button className="text-button" onClick={() => { setPrompt(''); resetResults(); }} disabled={!prompt && !analysis}>
                  <RotateCcw size={15} /> Clear
                </button>
              </div>
            </div>
            <div className="analyze-row">
              <button className="primary-button analyze-button" onClick={() => void handleAnalyze()} disabled={loadingAnalyze || loadingImprove || !prompt.trim()}>
                {loadingAnalyze ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={17} />}
                {loadingAnalyze ? 'Analyzing prompt…' : 'Analyze prompt'}
                {!loadingAnalyze && <ArrowRight size={17} />}
              </button>
              <span className="keyboard-hint"><kbd>⌘</kbd> <span>/</span> <kbd>Ctrl</kbd> <span>+</span> <kbd>Enter</kbd></span>
            </div>
            {error && <div className="error-alert" role="alert"><CircleHelp size={17} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={15} /></button></div>}
            <div className="examples-area">
              <span className="examples-label">TRY AN EXAMPLE</span>
              <div className="example-list">
                {EXAMPLES.map((example) => {
                  const Icon = example.icon;
                  return <button key={example.label} className="example-chip" onClick={() => loadExample(example.prompt)}><Icon size={14} />{example.label}<ArrowUpRight size={12} /></button>;
                })}
              </div>
            </div>
            <div className="privacy-note"><ShieldCheck size={15} /><span>Your prompt is processed for this request and is not saved to an application history.</span></div>
          </div>

          <aside className="side-column">
            {!analysis && !loadingAnalyze ? (
              <div className="side-intro surface-card">
                <div className="side-illustration">
                  <div className="orbit orbit-a" /><div className="orbit orbit-b" />
                  <div className="illustration-core"><WandSparkles size={29} /></div>
                  <span className="float-tag tag-a"><Target size={14} /> Intent</span>
                  <span className="float-tag tag-b"><Zap size={14} /> Clarity</span>
                  <span className="float-tag tag-c"><Check size={13} /> Structure</span>
                </div>
                <h3>Small changes.<br />More useful answers.</h3>
                <p>Get a clear picture of your prompt before you put it to work.</p>
                <div className="mini-feature"><span className="mini-feature-icon"><BarChart3 size={16} /></span><span><strong>See the full picture</strong><small>Understand 10 important dimensions.</small></span></div>
                <div className="mini-feature"><span className="mini-feature-icon"><Lightbulb size={16} /></span><span><strong>Know what to change</strong><small>Get specific, ranked suggestions.</small></span></div>
                <div className="mini-feature"><span className="mini-feature-icon"><WandSparkles size={16} /></span><span><strong>Take a better prompt away</strong><small>Copy or download your improved version.</small></span></div>
              </div>
            ) : loadingAnalyze ? (
              <div className="loading-card surface-card"><div className="loading-orbit"><LoaderCircle className="spin" size={27} /></div><h3>Reading between the lines</h3><p>Checking task clarity, context, specificity, and the other dimensions.</p><div className="skeleton-line"/><div className="skeleton-line short"/></div>
            ) : analysis ? (
              <div className="result-summary surface-card">
                <div className="card-heading-row summary-title-row"><div><div className="section-kicker">YOUR PROMPT SCORE</div><h2>At a glance</h2></div><span className="live-dot"><span /> Analysis ready</span></div>
                <div className="score-summary-layout">
                  <ScoreRing score={analysis.score} large />
                  <div className="score-summary-copy"><span className={`category-pill ${scoreColor(analysis.score)}`}>{analysis.category}</span><h3>{analysis.assessment}</h3><p>
                   Score is an estimate combining the explainable prompt rubric
                   with the ML signal shown in the scoring methodology.
                   </p></div>
                </div>
                <div className="score-mini-stats">
                  <div><span>Rubric score</span><strong>{Math.round(analysis.rubric_score)}<small>/100</small></strong></div>
                  <div><span>Prompt length</span><strong>{String(analysis.features.word_count ?? wordCount)}<small> words</small></strong></div>
                  <div><span>Model</span><strong className="model-version">{shortModelName(analysis.model.version)}</strong></div>
                </div>
                <button className="method-toggle" onClick={() => setShowMethod(!showMethod)} aria-expanded={showMethod}>{showMethod ? 'Hide score methodology' : 'How was this scored?'}<ChevronDown size={15} className={showMethod ? 'rotate-chevron' : ''} /></button>

{showMethod && (
  <div className="method-explainer">
    {analysis.model.scoring_mode === 'colab_regression' ? (
      <>
        <p>
          <strong>Colab regression (60%)</strong> predicts a
          score from the dataset's proxy-quality labels.
          <strong> Rubric (40%)</strong> evaluates visible
          prompt characteristics across ten dimensions.
        </p>
        <p>
          The training labels are model-generated proxies,
          not independent human ground truth or a guarantee
          of downstream answer quality.
        </p>
      </>
    ) : analysis.model.scoring_mode ===
      'bpo_preference_classifier' ? (
      <>
        <p>
          <strong>Rubric (80%)</strong> evaluates prompt
          characteristics. <strong>BPO preference signal (20%)</strong>
          estimates similarity to the optimized-style class.
        </p>
        <p>
          This is a relative-preference signal, not a
          universal prompt-quality score.
        </p>
      </>
    ) : (
      <>
        <p>
          <strong>Rubric-only mode:</strong> the saved ML
          model was unavailable or could not make a prediction.
          The current score comes from the explainable rubric.
        </p>
        <p>
          Check the backend model status before relying on
          the ML component.
        </p>
      </>
    )}
  </div>
)}
                <div className="summary-action-row"><span><CheckCircle2 size={15} /> Analysis ready</span><button className="text-button" onClick={resetResults}><RotateCcw size={14} /> Start over</button></div>
              </div>
            ) : null}
          </aside>
        </section>

        {analysis && (
          <section className="results-section" aria-label="Prompt analysis results">
            <div className="results-header"><div><div className="section-kicker">THE DETAILS</div><h2>Where your prompt stands</h2><p>Every score is an estimate. The dimensions point to useful edits, not universal rules.</p></div><button className="outline-button" onClick={() => void handleAnalyze()} disabled={loadingAnalyze}><RotateCcw size={15} /> Re-analyze</button></div>
            <div className="results-grid">
              <div className="surface-card dimensions-card"><div className="card-heading-row"><div><h3>Quality breakdown</h3><p>Ten aspects of a useful prompt</p></div><span className="ten-badge">10 dimensions</span></div><DimensionBars analysis={analysis} /><div className="dimension-footnote"><CircleHelp size={14} /><span>A lower score can indicate an opportunity—not every dimension is essential for every task.</span></div></div>
              <div className="insight-column">
                <div className="surface-card insight-card"><div className="insight-heading"><span className="insight-icon positive"><CheckCircle2 size={18} /></span><div><h3>What's working</h3><p>Keep these strengths</p></div></div><ul className="insight-list">{analysis.strengths.map((item, index) => <li key={`${item}-${index}`}><Check size={14} />{item}</li>)}</ul></div>
                <div className="surface-card insight-card"><div className="insight-heading"><span className="insight-icon attention"><Target size={18} /></span><div><h3>Room to improve</h3><p>Focus on the biggest gaps first</p></div></div>{analysis.weaknesses.length ? <ul className="insight-list">{analysis.weaknesses.map((item, index) => <li key={`${item}-${index}`}><ArrowUpRight size={14} />{item}</li>)}</ul> : <p className="empty-insight">No major weaknesses were flagged by the current rubric.</p>}</div>
              </div>
            </div>
            <div className="surface-card recommendations-card"><div className="recommendation-title"><span className="recommendation-symbol"><Lightbulb size={20} /></span><div><h3>Make the next edit count</h3><p>Suggestions based on this prompt—not a generic checklist.</p></div></div><div className="recommendation-list">{analysis.recommendations.map((item, index) => <div className="recommendation-item" key={`${item}-${index}`}><span className="recommendation-index">{String(index + 1).padStart(2, '0')}</span><span>{item}</span><ArrowRight size={15} /></div>)}</div></div>

            <div className="improve-section-heading"><div><div className="section-kicker">PUT IT INTO PRACTICE</div><h2>Build a clearer version</h2><p>Preserve your goal, make important unknowns visible, and keep the result focused.</p></div><label className="provider-toggle"><input type="checkbox" checked={useProvider} onChange={(event) => setUseProvider(event.target.checked)} /><span className="toggle-track" /><span>Use configured LLM provider <small>optional · may send prompt text externally</small></span></label></div>
            <div className="surface-card improve-card">
              {!improvement ? (
                <div className="improve-empty"><div className="improve-empty-icon"><WandSparkles size={26} /></div><div><h3>Turn the feedback into a prompt you can reuse</h3><p>{useProvider ? 'Uses the provider configured on the backend. Provider errors are shown; your analysis is preserved.' : 'Uses a deterministic, no-API-key rewrite fallback with placeholders instead of invented context.'}</p></div><button className="primary-button" onClick={() => void handleImprove()} disabled={loadingImprove}>{loadingImprove ? <LoaderCircle className="spin" size={17} /> : <Sparkles size={17} />}{loadingImprove ? 'Improving…' : 'Improve prompt'}<ArrowRight size={16} /></button></div>
              ) : improvement.status !== 'success' ? (
                <div className="provider-error"><CircleHelp size={21} /><div><h3>Rewrite not completed</h3><p>{improvement.error || 'The configured provider did not return a usable rewrite.'} Your original analysis is still available.</p></div><button className="outline-button" onClick={() => void handleImprove()} disabled={loadingImprove}><RotateCcw size={14} /> Retry</button><button className="text-button" onClick={() => { setUseProvider(false); setImprovement(null); setError(''); void handleImprove(false); }} disabled={loadingImprove}>Use fallback instead</button></div>
              ) : (
                <div className="improved-results">
                  <div className="improved-results-top"><div><span className="completed-label"><CheckCircle2 size={15} /> IMPROVEMENT READY</span><h3>Compare before and after</h3><p>{improvement.notice || `Generated with ${improvement.provider || 'configured provider'}. Review the rewrite before using it.`}</p></div><button className="text-button" onClick={() => { setImprovement(null); setEditableRewrite(null); setRewriteEditDirty(false); setCopied(false); }}><RotateCcw size={14} /> New rewrite</button></div>
                  <div className="compare-grid">
                    <div className="compare-panel original-panel"><div className="compare-panel-heading"><span className="compare-dot original-dot" /> Original prompt <ScorePill score={analysis.score} /></div><div className="compare-copy">{prompt}</div></div>
                    <div className="compare-divider"><ArrowRight size={17} /></div>
                    <div className="compare-panel improved-panel"><div className="compare-panel-heading"><span className="compare-dot improved-dot" /> Improved prompt {rewriteEditDirty ? <span className="edited-score-label">Edited · score needs refresh</span> : improvement.improved_analysis && <ScorePill score={improvement.improved_analysis.score} />}</div><label htmlFor="improved-prompt" className="sr-only">Edit improved prompt</label><textarea id="improved-prompt" className="improved-prompt-textarea" value={rewrittenText} onChange={(event) => { setEditableRewrite(event.target.value); setRewriteEditDirty(true); setCopied(false); }} />
                      <div className="rewrite-actions"><button className={`primary-button copy-button ${copied ? 'is-copied' : ''}`} onClick={() => void copyRewrite()} disabled={!rewrittenText.trim()}>{copied ? <Check size={15} /> : <Copy size={15} />}{copied ? 'Copied' : 'Copy prompt'}</button><button className="outline-button download-button" onClick={downloadRewrite} disabled={!rewrittenText.trim()}><Download size={15} /> Download .txt</button>{rewriteEditDirty && <button className="outline-button rescore-button" onClick={() => void rescoreEditedRewrite()} disabled={recheckingRewrite || !rewrittenText.trim()}>{recheckingRewrite ? <LoaderCircle className="spin" size={14} /> : <BarChart3 size={14} />}{recheckingRewrite ? 'Re-scoring…' : 'Re-score edit'}</button>}</div>
                    </div>
                  </div>
                  {improvement.improved_analysis && !rewriteEditDirty && <div className="score-change-row"><span className="score-change-icon">{(improvement.score_delta ?? 0) >= 0 ? <ArrowUpRight size={17} /> : <ArrowDown size={17} />}</span><div><strong>{improvement.score_delta === undefined || improvement.score_delta === null ? 'Score comparison unavailable' : `${improvement.score_delta > 0 ? '+' : ''}${Math.round(improvement.score_delta)} points`}</strong><p>{(improvement.score_delta ?? 0) > 0 ? 'The rubric estimates a stronger structure for the revised text. This does not guarantee better real-world model output.' : (improvement.score_delta ?? 0) < 0 ? 'The current scoring method did not estimate a higher score. Review the changes; do not add text just to chase a score.' : 'The scoring method estimates similar quality. A higher score is not guaranteed or artificially forced.'}</p></div></div>}
                  {!!improvement.changes.length && <div className="changes-made"><strong>What changed</strong><ul>{improvement.changes.map((change, index) => <li key={`${change}-${index}`}>{change}</li>)}</ul></div>}
                </div>
              )}
            </div>
            <p className="results-caveat"><ShieldCheck size={14} /> Promptly estimates prompt quality from surface features and learned relative preferences. Different LLMs and tasks can respond differently; review rewritten prompts before relying on them.</p>
          </section>
        )}

        <section id="how-it-works" className="how-section">
          <div className="how-heading"><div className="section-kicker">A LITTLE MORE INTENTION</div><h2>Good prompts aren't necessarily longer.</h2><p>They make the important parts easier for an AI to understand.</p></div>
          <div className="how-grid">
            <div className="how-step surface-card"><span className="step-number">01</span><div className="how-step-icon"><Target size={21} /></div><h3>Understand the task</h3><p>We check the request, scope, clarity, and how actionable the instructions are.</p></div>
            <div className="how-step surface-card"><span className="step-number">02</span><div className="how-step-icon"><BarChart3 size={21} /></div><h3>Find meaningful gaps</h3><p>See where context, constraints, audience, or format could help—when they're relevant.</p></div>
            <div className="how-step surface-card"><span className="step-number">03</span><div className="how-step-icon"><WandSparkles size={21} /></div><h3>Refine with intent</h3><p>Get a model-agnostic version that makes unknowns explicit instead of making things up.</p></div>
          </div>
        </section>

        <footer className="footer"><a className="brand footer-brand" href="#top"><span className="brand-mark"><Sparkles size={15} /></span><span>promptly<span className="brand-dot">.</span></span></a><span>Built to make your next question clearer.</span><div className="footer-right"><span><FlaskConical size={14} /> Research prototype</span><a href="#how-it-works">Scoring notes <ArrowUpRight size={12} /></a></div></footer>
      </main>
    </div>
  );
}
