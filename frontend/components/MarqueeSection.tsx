const ITEMS = [
  '✓ Urdu Fake News Detection',
  '✦ xlm-RoBERTa Fine-Tuned',
  '✓ Binary Classification',
  '✦ REAL / FAKE Verdict',
  '✓ CPU-Only Inference',
  '✦ Real-Time Analysis',
  '✓ Prediction History',
  '✦ سچ کی پہچان',
  '✓ Urdu Text Preprocessing',
  '✦ Confidence Score',
  '✓ FastAPI Backend',
  '✦ Next.js Frontend',
]

// Duplicate for seamless loop
const DOUBLED = [...ITEMS, ...ITEMS]

export default function MarqueeSection() {
  return (
    <section className="py-12 overflow-hidden border-y border-white/5">
      <div className="marquee-wrapper">
        <div className="marquee">
          {DOUBLED.map((item, i) => (
            <span
              key={i}
              className={`flex-shrink-0 text-sm font-medium px-2 ${
                item.startsWith('✓')
                  ? 'text-emerald-400'
                  : item.includes('سچ')
                  ? 'text-purple-300 urdu'
                  : 'text-slate-500'
              }`}
            >
              {item}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
