import { Activity } from 'lucide-react';
import { motion } from 'motion/react';

export default function LoadingState({ message = 'Analyzing your symptoms...' }) {
  return (
    <div className="rounded-[28px] border border-white/10 bg-[#0d0d0d]/90 p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="flex items-center gap-3 text-slate-200">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        >
          <Activity className="h-5 w-5" />
        </motion.div>
        <p className="font-medium">{message}</p>
      </div>
    </div>
  );
}
