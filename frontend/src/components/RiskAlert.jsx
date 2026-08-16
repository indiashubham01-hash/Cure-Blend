import { ShieldAlert } from 'lucide-react';

export default function RiskAlert({ emergency }) {
  if (!emergency || !emergency.is_emergency) {
    return null;
  }

  return (
    <div className="rounded-[28px] border border-red-500/30 bg-red-500/10 p-5 text-red-50 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="flex items-start gap-3">
        <ShieldAlert className="mt-0.5 h-5 w-5 text-red-300" />
        <div>
          <p className="text-lg font-semibold">Emergency alert</p>
          <p className="mt-1 text-sm text-red-100">{emergency.advisory}</p>
          {emergency.matched_flags?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {emergency.matched_flags.map((flag) => (
                <span key={flag} className="rounded-full border border-red-300/30 bg-red-500/10 px-2.5 py-1 text-xs font-medium text-red-100">
                  {flag}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
