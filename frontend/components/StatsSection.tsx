import { Brain, Cpu, Languages, ShieldCheck } from 'lucide-react'

const STAT_CARDS = [
  { icon: Brain,       label: 'Model',          value: 'xlm-RoBERTa', color: '#a78bfa' },
  { icon: Languages,   label: 'Language',        value: 'Urdu',        color: '#22d3ee' },
  { icon: Cpu,         label: 'Inference',       value: 'CPU-Only',    color: '#34d399' },
  { icon: ShieldCheck, label: 'Classification',  value: 'REAL / FAKE', color: '#fbbf24' },
]

export default function StatsSection() {
  return (
    <section id="stats" className="relative py-24 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-14">
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            Model <span className="gradient-text">Overview</span>
          </h2>
          <p className="text-slate-400 max-w-lg mx-auto">
            Fine-tuned xlm-RoBERTa for Urdu fake news binary classification
          </p>
        </div>

        {/* Stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
          {STAT_CARDS.map((s, i) => (
            <div key={i} className="glass rounded-2xl p-5 text-center feature-card">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-3"
                style={{ background: `${s.color}22`, border: `1px solid ${s.color}33` }}
              >
                <s.icon className="w-5 h-5" style={{ color: s.color }} />
              </div>
              <div className="text-xl font-extrabold text-white mb-1">{s.value}</div>
              <div className="text-xs text-slate-500">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Architecture tags */}
        <div className="glass rounded-xl p-5 flex flex-wrap gap-3 items-center justify-center text-xs text-slate-500">
          {[
            'xlm-roberta-base',
            '768-dim embeddings',
            '12 attention layers',
            'Max 128 tokens',
            'Float32 precision',
            'HuggingFace Transformers',
            'FastAPI backend',
            'aiosqlite storage',
          ].map((tag) => (
            <span key={tag} className="px-3 py-1 rounded-full border border-white/10 text-slate-400">
              {tag}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
