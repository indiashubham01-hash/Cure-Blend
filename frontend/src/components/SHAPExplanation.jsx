export default function SHAPExplanation({ items }) {
  if (!items || items.length === 0) {
    return (
      <div className="rounded-[28px] border border-white/10 bg-[#0d0d0d]/80 p-5 text-slate-300 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
        SHAP explanation data is unavailable for this assessment.
      </div>
    );
  }

  const maxWeight = Math.max(...items.map((item) => Math.abs(item.weight || 0)), 1);

  return (
    <div className="rounded-[28px] border border-white/10 bg-[#0d0d0d]/80 p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <h3 className="text-lg font-semibold text-white">Feature importance</h3>
      <div className="mt-4 space-y-3">
        {items.map((item) => (
          <div key={item.feature}>
            <div className="mb-1 flex items-center justify-between gap-3 text-sm">
              <span className="font-medium text-slate-200">{item.feature}</span>
              <span className="text-slate-400">{Number(item.weight || 0).toFixed(3)}</span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-[#1a1a1a]">
              <div
                className="h-full rounded-full bg-white"
                style={{ width: `${Math.max(8, (Math.abs(item.weight || 0) / maxWeight) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
