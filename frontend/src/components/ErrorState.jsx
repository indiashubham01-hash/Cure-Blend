import { AlertTriangle } from 'lucide-react';

export default function ErrorState({ message }) {
  return (
    <div className="rounded-[28px] border border-rose-500/30 bg-rose-500/10 p-5 text-rose-100 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 flex-none text-rose-300" />
        <div>
          <p className="font-semibold">Unable to complete assessment</p>
          <p className="mt-1 text-sm text-rose-200">{message}</p>
        </div>
      </div>
    </div>
  );
}
