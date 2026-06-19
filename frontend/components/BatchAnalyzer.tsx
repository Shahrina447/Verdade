'use client'

import { useState, useCallback } from 'react'
import { Plus, Trash2, Layers, RefreshCw, CheckCircle, AlertTriangle, Loader2 } from 'lucide-react'
import { analyzeNewsBatch, PredictResponse, BatchSummary } from '@/lib/api'

// ── Types ─────────────────────────────────────────────────────────────────────
interface ArticleRow {
  id: number
  text: string
}

interface BatchResult {
  results: PredictResponse[]
  summary: BatchSummary
}

// ── Helpers ───────────────────────────────────────────────────────────────────
let nextId = 1
function makeRow(text = ''): ArticleRow {
  return { id: nextId++, text }
}

function VerdictBadge({ prediction }: { prediction: 'FAKE' | 'REAL' }) {
  if (prediction === 'REAL') {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold badge-real">
        <CheckCircle className="w-3.5 h-3.5" />
        REAL
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold badge-fake">
      <AlertTriangle className="w-3.5 h-3.5" />
      FAKE
    </span>
  )
}

// ── Summary Card ──────────────────────────────────────────────────────────────
function SummaryCard({
  label,
  value,
  color,
  sublabel,
}: {
  label: string
  value: number | string
  color: string
  sublabel?: string
}) {
  return (
    <div className="glass rounded-xl p-5 flex flex-col items-center justify-center gap-1 text-center">
      <span className="text-3xl font-extrabold" style={{ color }}>
        {value}
      </span>
      <span className="text-sm font-semibold text-slate-300">{label}</span>
      {sublabel && <span className="text-xs text-slate-500">{sublabel}</span>}
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────────
export default function BatchAnalyzer() {
  const [rows, setRows] = useState<ArticleRow[]>([makeRow(), makeRow()])
  const [state, setState] = useState<'idle' | 'loading' | 'result'>('idle')
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const MAX_ROWS = 10

  // ── Row management ──────────────────────────────────────────────────────────
  const addRow = useCallback(() => {
    setRows((prev) => (prev.length < MAX_ROWS ? [...prev, makeRow()] : prev))
  }, [])

  const deleteRow = useCallback((id: number) => {
    setRows((prev) => {
      if (prev.length <= 1) return prev
      return prev.filter((r) => r.id !== id)
    })
  }, [])

  const updateRow = useCallback((id: number, text: string) => {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, text } : r)))
  }, [])

  // ── Analysis ────────────────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    const texts = rows.map((r) => r.text.trim()).filter((t) => t.length >= 10)
    if (texts.length === 0) {
      setError('Please enter at least one article with 10+ characters.')
      return
    }

    setState('loading')
    setError(null)

    try {
      const data = await analyzeNewsBatch(texts)
      setBatchResult(data)
      setState('result')
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : 'Backend unreachable. Please ensure the server is running on port 8000.'
      setError(msg)
      setState('idle')
    }
  }

  const handleReset = () => {
    setState('idle')
    setBatchResult(null)
    setError(null)
    setRows([makeRow(), makeRow()])
  }

  const filledCount = rows.filter((r) => r.text.trim().length >= 10).length

  return (
    <section id="batch" className="relative py-24 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-xs text-cyan-300 mb-4 border border-cyan-500/20">
            <Layers className="w-3.5 h-3.5" />
            <span>Batch Analysis</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-3">
            Analyze Multiple{' '}
            <span className="gradient-text">Articles at Once</span>
          </h2>
          <p className="urdu text-slate-400 text-xl">ایک ساتھ کئی خبروں کا تجزیہ کریں</p>
        </div>

        {/* Card */}
        <div className="glass rounded-2xl p-6 sm:p-8" style={{ boxShadow: '0 0 80px rgba(6, 182, 212, 0.15)' }}>
          {/* ── INPUT STATE ── */}
          {state !== 'result' && (
            <>
              {/* Article rows */}
              <div className="space-y-3 mb-5">
                {rows.map((row, idx) => (
                  <div key={row.id} className="flex items-start gap-3">
                    {/* Row number */}
                    <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-slate-500 border border-white/10 mt-2">
                      {idx + 1}
                    </div>

                    {/* Textarea */}
                    <textarea
                      value={row.text}
                      onChange={(e) => updateRow(row.id, e.target.value)}
                      rows={2}
                      dir="rtl"
                      lang="ur"
                      placeholder={`یہاں خبر ${idx + 1} پیسٹ کریں...`}
                      className="urdu flex-1 bg-transparent border border-white/10 rounded-xl px-4 py-3 text-slate-200 placeholder:text-slate-600 text-base resize-none focus:outline-none focus:border-cyan-500/50 transition-colors leading-loose"
                    />

                    {/* Delete button — always visible but disabled when only 1 row */}
                    <button
                      onClick={() => deleteRow(row.id)}
                      disabled={rows.length <= 1}
                      aria-label={`Remove article ${idx + 1}`}
                      className="flex-shrink-0 mt-2 p-2 rounded-lg glass border border-white/5 text-slate-500 hover:text-red-400 hover:border-red-500/30 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>

              {/* Add article button */}
              {rows.length < MAX_ROWS && (
                <button
                  onClick={addRow}
                  className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-dashed border-white/15 text-slate-500 text-sm hover:border-cyan-500/40 hover:text-cyan-400 transition-colors mb-5"
                >
                  <Plus className="w-4 h-4" />
                  Add Article
                  <span className="text-xs opacity-60">({rows.length}/{MAX_ROWS})</span>
                </button>
              )}

              {/* Error */}
              {error && (
                <div className="mb-4 px-4 py-3 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-sm">
                  {error}
                </div>
              )}

              {/* Analyze button */}
              <button
                onClick={handleAnalyze}
                disabled={state === 'loading' || filledCount === 0}
                className="w-full btn-primary px-6 py-3.5 rounded-xl text-white font-semibold flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ background: 'linear-gradient(135deg, #0891b2, #7c3aed)' }}
              >
                {state === 'loading' ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Analyzing {filledCount} articles…</span>
                  </>
                ) : (
                  <>
                    <Layers className="w-4 h-4" />
                    <span>
                      Analyze All
                      {filledCount > 0 && (
                        <span className="ml-1.5 text-xs opacity-75">({filledCount} article{filledCount !== 1 ? 's' : ''})</span>
                      )}
                    </span>
                  </>
                )}
              </button>

              {/* Loading scan */}
              {state === 'loading' && (
                <div className="relative mt-5 glass rounded-xl h-16 overflow-hidden flex items-center justify-center">
                  <div className="scan-line" />
                  <p className="text-xs text-slate-500 z-10">Running batch xlm-RoBERTa inference…</p>
                </div>
              )}
            </>
          )}

          {/* ── RESULT STATE ── */}
          {state === 'result' && batchResult && (
            <div className="space-y-8">
              {/* Summary cards */}
              <div>
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
                  Batch Summary
                </h3>
                <div className="grid grid-cols-3 gap-4">
                  <SummaryCard
                    label="Total"
                    value={batchResult.summary.total}
                    color="#a78bfa"
                    sublabel="articles"
                  />
                  <SummaryCard
                    label="Real"
                    value={batchResult.summary.real}
                    color="#34d399"
                    sublabel={`${(100 - batchResult.summary.fake_percentage).toFixed(1)}%`}
                  />
                  <SummaryCard
                    label="Fake"
                    value={batchResult.summary.fake}
                    color="#f87171"
                    sublabel={`${batchResult.summary.fake_percentage.toFixed(1)}%`}
                  />
                </div>

                {/* Fake percentage bar */}
                <div className="mt-4 space-y-1.5">
                  <div className="flex justify-between text-xs text-slate-500">
                    <span className="text-emerald-400 font-medium">
                      Real — {(100 - batchResult.summary.fake_percentage).toFixed(1)}%
                    </span>
                    <span className="text-red-400 font-medium">
                      Fake — {batchResult.summary.fake_percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden bg-white/5 flex">
                    <div
                      className="h-full rounded-l-full transition-all duration-1000"
                      style={{
                        width: `${100 - batchResult.summary.fake_percentage}%`,
                        background: '#34d399',
                      }}
                    />
                    <div
                      className="h-full rounded-r-full"
                      style={{
                        width: `${batchResult.summary.fake_percentage}%`,
                        background: '#f87171',
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Results table */}
              <div>
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
                  Individual Results
                </h3>
                <div className="rounded-xl overflow-hidden border border-white/8">
                  {/* Table header */}
                  <div className="grid grid-cols-[2rem_1fr_auto_auto] gap-4 px-4 py-3 bg-white/5 text-xs font-semibold text-slate-500 uppercase tracking-widest">
                    <span>#</span>
                    <span className="text-right">Text Preview</span>
                    <span>Verdict</span>
                    <span>Confidence</span>
                  </div>

                  {/* Rows */}
                  {batchResult.results.map((res, idx) => {
                    const originalText = rows[idx]?.text ?? ''
                    const preview = originalText.trim().slice(0, 60) + (originalText.trim().length > 60 ? '…' : '')
                    return (
                      <div
                        key={res.prediction_id}
                        className="grid grid-cols-[2rem_1fr_auto_auto] gap-4 px-4 py-4 border-t border-white/5 items-center hover:bg-white/3 transition-colors"
                      >
                        {/* # */}
                        <span className="text-xs text-slate-600 font-mono">{idx + 1}</span>

                        {/* Text preview — RTL */}
                        <p
                          dir="rtl"
                          lang="ur"
                          className="urdu text-slate-300 text-sm leading-relaxed text-right truncate"
                          title={originalText.trim()}
                        >
                          {preview}
                        </p>

                        {/* Verdict badge */}
                        <VerdictBadge prediction={res.prediction as 'FAKE' | 'REAL'} />

                        {/* Confidence */}
                        <span
                          className="text-sm font-bold tabular-nums"
                          style={{ color: res.prediction === 'REAL' ? '#34d399' : '#f87171' }}
                        >
                          {Math.round(res.confidence * 100)}%
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-3">
                <button
                  onClick={handleReset}
                  className="flex-1 flex items-center justify-center gap-2 px-5 py-3 rounded-xl glass border border-white/10 text-slate-300 text-sm font-medium hover:border-cyan-500/30 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  New Batch
                </button>
                <a
                  href="/history"
                  className="flex-1 flex items-center justify-center gap-2 px-5 py-3 rounded-xl text-white text-sm font-medium"
                  style={{ background: 'linear-gradient(135deg, #0891b2, #7c3aed)' }}
                >
                  View History
                </a>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
