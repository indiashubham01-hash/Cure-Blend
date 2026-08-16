import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import {
  ArrowRight,
  BellRing,
  BrainCircuit,
  CheckCircle2,
  HeartPulse,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  TriangleAlert,
} from 'lucide-react';
import { BrowserRouter, Link, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { assessHealth, getHealthStatus } from './lib/api';
import { commonConditions, commonLifestyleFactors, defaultSymptomOptions } from './data/symptomOptions';
import ErrorState from './components/ErrorState';
import LoadingState from './components/LoadingState';
import RiskAlert from './components/RiskAlert';
import SHAPExplanation from './components/SHAPExplanation';

const initialForm = {
  symptoms: ['fever'],
  symptom_text: 'I have fever and mild cough',
  age: 32,
  lifestyle_factors: ['stress'],
  existing_conditions: ['hypertension'],
};

const featureCards = [
  {
    icon: BrainCircuit,
    title: 'AI risk analysis',
    text: 'Model-driven symptom evaluation across common conditions and health patterns.',
  },
  {
    icon: ShieldCheck,
    title: 'Safety-first checks',
    text: 'Emergency flags are reviewed before standard recommendations are generated.',
  },
  {
    icon: HeartPulse,
    title: 'Personalized guidance',
    text: 'Recommendations adapt to age, conditions, and lifestyle factors from the backend.',
  },
];

const steps = ['Describe symptoms', 'Review assessment', 'View risk and SHAP', 'Check recommendations'];

const panelClass = 'rounded-[28px] border border-white/10 bg-[#111111] shadow-[0_0_0_1px_rgba(255,255,255,0.04),0_18px_40px_rgba(0,0,0,0.45)]';
const softButton = 'inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white px-4 py-2.5 text-sm font-medium text-black transition hover:bg-[#f3f3f3]';

function Layout({ children }) {
  return (
    <div className="min-h-screen bg-[#050505] text-white">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.05),transparent_30%)]" />
      <header className="sticky top-0 z-40 border-b border-white/10 bg-[#090909]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1280px] items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
          <Link to="/" className="flex items-center gap-3 font-semibold text-white">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-white shadow-[0_0_18px_rgba(255,255,255,0.08)]">
              <HeartPulse className="h-4 w-4" />
            </div>
            <div className="text-base tracking-[0.18em] text-white">CURE-BLEND</div>
          </Link>

          <nav className="hidden items-center gap-7 text-sm text-zinc-300 md:flex">
            <Link to="/" className="transition hover:text-white">Home</Link>
            <Link to="/assessment" className="transition hover:text-white">Assessment</Link>
            <Link to="/results" className="transition hover:text-white">Results</Link>
            <Link to="/risk" className="transition hover:text-white">Risk</Link>
            <Link to="/shap" className="transition hover:text-white">SHAP</Link>
            <Link to="/medical-info" className="transition hover:text-white">Medical Info</Link>
          </nav>

          <div className="flex items-center gap-3">
            <button className="hidden rounded-full border border-white/10 bg-[#111111] p-2 text-zinc-200 hover:bg-[#171717] md:inline-flex">
              <BellRing className="h-4 w-4" />
            </button>
            <button className="hidden rounded-full border border-white/10 bg-[#111111] p-2 text-zinc-200 hover:bg-[#171717] md:inline-flex">
              <ShieldCheck className="h-4 w-4" />
            </button>
            <Link to="/assessment" className="rounded-full border border-white/10 bg-[#111111] px-4 py-2 text-sm text-white hover:bg-[#171717]">
              Dashboard
            </Link>
          </div>
        </div>
      </header>
      <div className="relative">{children}</div>
    </div>
  );
}

function LandingPage() {
  return (
    <main className="relative overflow-hidden">
      <section className="relative overflow-hidden border-b border-white/10 bg-[#050505]">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(255,255,255,0.09),transparent_28%),radial-gradient(circle_at_top,_rgba(255,255,255,0.05),transparent_24%),radial-gradient(circle_at_center,_rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:100%_100%,100%_100%,18px_18px] opacity-90" />
        <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-gradient-to-b from-white/[0.03] via-transparent to-transparent" />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-[#050505] via-[#050505]/80 to-transparent" />

        <div className="relative mx-auto max-w-[1280px] px-4 pb-14 pt-10 sm:px-6 lg:px-8 lg:pb-20 lg:pt-14">
          <div className="mx-auto max-w-5xl text-center">
            <h1 className="mx-auto max-w-5xl text-5xl font-semibold tracking-[-0.07em] text-white sm:text-6xl lg:text-[7rem] lg:leading-[0.86]">
              CURE-BLEND
              <span className="block text-zinc-300">Smarter health insights</span>
              <span className="block text-zinc-100">for better care decisions</span>
            </h1>

            <p className="mx-auto mt-7 max-w-3xl text-lg leading-8 text-zinc-400">
              CURE-BLEND helps patients and care teams turn symptoms, lifestyle context, and risk signals into clearer, more confident next steps for health decisions.
            </p>

          </div>

          <div className="relative mx-auto mt-14 max-w-6xl rounded-[28px] border border-white/10 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.08),transparent_35%),linear-gradient(180deg,#0b0b0b_0%,#070707_100%)] p-2 shadow-[0_30px_80px_rgba(0,0,0,0.78)]">
            <div className="rounded-[22px] border border-white/10 bg-[#0c0c0c] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
              <div className="flex items-center justify-between gap-3 border-b border-white/10 pb-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-[#111111] text-white">
                    <HeartPulse className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-xl font-semibold text-white">CURE-BLEND</div>
                    <div className="text-[10px] uppercase tracking-[0.22em] text-zinc-500">Health dashboard</div>
                  </div>
                </div>

                <div className="flex items-center gap-2 rounded-full border border-white/10 bg-[#111111] px-3 py-1.5 text-sm text-zinc-300">
                  <span className="h-2.5 w-2.5 rounded-full bg-white/80" />
                  <span>Search...</span>
                </div>
              </div>

              <div className="mt-5 grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
                <aside className="rounded-[20px] border border-white/10 bg-[#0d0d0d] p-4">
                  <div className="mb-5 flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                    <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-white/5 text-zinc-200">
                      <HeartPulse className="h-3.5 w-3.5" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-white">CURE-BLEND</div>
                      <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-500">Admin</div>
                    </div>
                  </div>

                  <div className="space-y-2 text-sm text-zinc-300">
                    {['Overview', 'Risk Analysis', 'Patient Trends', 'Assessments', 'Medication Review', 'Alerts', 'Clinical Notes', 'Reports', 'Care Team'].map((item, index) => (
                      <div
                        key={item}
                        className={`flex items-center gap-3 rounded-xl px-3 py-2 ${index === 0 ? 'bg-white/5 text-white' : 'text-zinc-300'}`}
                      >
                        <span className="inline-block h-2 w-2 rounded-full bg-zinc-500" />
                        {item}
                      </div>
                    ))}
                  </div>
                </aside>

                <div className="rounded-[20px] border border-white/10 bg-[#0d0d0d] p-5">
                  <div className="flex items-center justify-between gap-3 pb-4">
                    <div>
                      <h2 className="text-4xl font-semibold tracking-[-0.06em] text-white">Overview</h2>
                      <p className="mt-2 text-zinc-400">Patient risk summary and care insights</p>
                    </div>
                    <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-zinc-200">
                      Live
                    </div>
                  </div>

                  <div className="mt-6 grid gap-4 md:grid-cols-4">
                    {[
                      { label: 'Risk Score', value: '72', trend: '+8.4%', note: 'Higher than last review' },
                      { label: 'Active Patients', value: '1,234', trend: '+12%', note: 'Growth in monitored cases' },
                      { label: 'Critical Alerts', value: '18', trend: '-6.2%', note: 'Improved since last cycle' },
                      { label: 'Recovery Rate', value: '84%', trend: '+4.9%', note: 'Improving patient outcomes' },
                    ].map((card) => (
                      <div key={card.label} className="rounded-2xl border border-white/10 bg-[#111111] p-4">
                        <div className="flex items-center justify-between text-sm text-zinc-400">
                          <span>{card.label}</span>
                          <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] uppercase tracking-[0.16em] text-zinc-300">{card.trend}</span>
                        </div>
                        <div className="mt-4 text-3xl font-semibold text-white">{card.value}</div>
                        <div className="mt-3 text-sm text-zinc-400">{card.note}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

function AssessmentPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState(initialForm);
  const [selectedSymptoms, setSelectedSymptoms] = useState(initialForm.symptoms);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const toggleSymptom = (symptom) => {
    const next = selectedSymptoms.includes(symptom)
      ? selectedSymptoms.filter((item) => item !== symptom)
      : [...selectedSymptoms, symptom];

    setSelectedSymptoms(next);
    setForm((current) => ({ ...current, symptoms: next }));
  };

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: name === 'age' ? Number(value) : value }));
  };

  const handleCheck = (group, value) => {
    const current = form[group] || [];
    const next = current.includes(value)
      ? current.filter((item) => item !== value)
      : [...current, value];

    setForm((currentForm) => ({ ...currentForm, [group]: next }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const payload = {
        ...form,
        symptoms: selectedSymptoms,
        symptom_text: form.symptom_text || selectedSymptoms.join(', '),
      };

      const { data } = await assessHealth(payload);
      navigate('/results', { state: { result: data } });
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || 'Assessment failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-6xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-8">
        <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-300">Assessment</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-white">Check symptoms and health context</h1>
      </div>

      <form onSubmit={handleSubmit} className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-8">
          <div className={`${panelClass} p-6`}>
            <label className="mb-3 block text-sm font-medium text-slate-200">Symptoms</label>
            <div className="flex flex-wrap gap-2">
              {defaultSymptomOptions.map((symptom) => {
                const isSelected = selectedSymptoms.includes(symptom);
                return (
                  <button
                    key={symptom}
                    type="button"
                    onClick={() => toggleSymptom(symptom)}
                    className={`rounded-full border px-3 py-2 text-sm transition ${
                      isSelected
                        ? 'border-white/20 bg-white text-black'
                        : 'border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:bg-white/10'
                    }`}
                  >
                    {symptom}
                  </button>
                );
              })}
            </div>
          </div>

          <div className={`${panelClass} p-6`}>
            <label htmlFor="symptom_text" className="mb-3 block text-sm font-medium text-slate-200">
              Symptom description
            </label>
            <textarea
              id="symptom_text"
              name="symptom_text"
              value={form.symptom_text}
              onChange={handleInputChange}
              rows={5}
              className="w-full rounded-2xl border border-white/10 bg-[#0a0a0a] px-4 py-3 text-slate-100 outline-none transition focus:border-white/30"
              placeholder="Describe your symptoms in your own words..."
            />
          </div>
        </div>

        <div className="space-y-8">
          <div className={`${panelClass} p-6`}>
            <label htmlFor="age" className="mb-3 block text-sm font-medium text-slate-200">
              Age
            </label>
            <input
              id="age"
              name="age"
              type="number"
              min="0"
              max="120"
              value={form.age}
              onChange={handleInputChange}
              className="w-full rounded-2xl border border-white/10 bg-[#0a0a0a] px-4 py-3 text-slate-100 outline-none transition focus:border-white/30"
            />
          </div>

          <div className={`${panelClass} p-6`}>
            <p className="mb-3 text-sm font-medium text-slate-200">Lifestyle factors</p>
            <div className="flex flex-wrap gap-2">
              {commonLifestyleFactors.map((factor) => (
                <button
                  key={factor}
                  type="button"
                  onClick={() => handleCheck('lifestyle_factors', factor)}
                  className={`rounded-full border px-3 py-2 text-sm ${
                    form.lifestyle_factors.includes(factor)
                      ? 'border-white/20 bg-white text-black'
                      : 'border-white/10 bg-white/5 text-slate-300'
                  }`}
                >
                  {factor}
                </button>
              ))}
            </div>
          </div>

          <div className={`${panelClass} p-6`}>
            <p className="mb-3 text-sm font-medium text-slate-200">Existing conditions</p>
            <div className="flex flex-wrap gap-2">
              {commonConditions.map((condition) => (
                <button
                  key={condition}
                  type="button"
                  onClick={() => handleCheck('existing_conditions', condition)}
                  className={`rounded-full border px-3 py-2 text-sm ${
                    form.existing_conditions.includes(condition)
                      ? 'border-white/20 bg-white text-black'
                      : 'border-white/10 bg-white/5 text-slate-300'
                  }`}
                >
                  {condition}
                </button>
              ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-white px-5 py-3.5 text-base font-medium text-black transition hover:bg-[#f1f1f1] disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isLoading ? 'Analyzing…' : 'Analyze Symptoms'}
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </form>

      <div className="mt-8">
        {isLoading && <LoadingState />}
        {error && <ErrorState message={error} />}
      </div>
    </main>
  );
}

function useAssessmentResult() {
  return useLocation().state?.result ?? null;
}

function ResultsPage() {
  const result = useAssessmentResult();

  if (!result) {
    return (
      <main className="mx-auto max-w-6xl px-4 py-12 sm:px-6 lg:px-8">
        <ErrorState message="No assessment result is available yet. Please complete the assessment first." />
      </main>
    );
  }

  const topPrediction = result.predictions?.[0];
  const severityLevel = result.severity?.level || 'Unknown';
  const confidence = Number(topPrediction?.probability || 0) * 100;

  return (
    <main className="mx-auto max-w-6xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-8 flex items-center justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-300">Results</p>
          <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-white">Assessment outcome</h1>
        </div>
        <div className="flex gap-3">
          <Link to="/risk" state={{ result }} className={softButton}>View risk</Link>
          <Link to="/shap" state={{ result }} className={softButton}>View SHAP</Link>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className={`${panelClass} p-6 lg:col-span-2`}>
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm text-slate-400">Top prediction</p>
              <h2 className="mt-2 text-2xl font-semibold text-white">{topPrediction?.condition || 'Unknown'}</h2>
            </div>
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-white">
              <CheckCircle2 className="h-6 w-6" />
            </div>
          </div>

          <div className="mt-8">
            <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
              <span>Confidence</span>
              <span>{confidence.toFixed(0)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-[#1a1a1a]">
              <div className="h-full rounded-full bg-white" style={{ width: `${Math.min(confidence, 100)}%` }} />
            </div>
          </div>

          <div className="mt-8 space-y-4">
            <h3 className="text-lg font-semibold text-white">Condition probability</h3>
            {result.predictions?.map((prediction) => (
              <div key={prediction.condition}>
                <div className="mb-1 flex items-center justify-between text-sm text-slate-300">
                  <span>{prediction.condition}</span>
                  <span>{(prediction.probability * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 overflow-hidden rounded-full bg-[#1a1a1a]">
                  <div className="h-full rounded-full bg-white" style={{ width: `${prediction.probability * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-6">
          <div className={`${panelClass} p-6`}>
            <p className="text-sm text-slate-400">Severity</p>
            <h3 className="mt-2 text-2xl font-semibold text-white">{severityLevel}</h3>
            <p className="mt-3 text-sm text-slate-300">Score: {result.severity?.score ?? 0}/100</p>
          </div>

          <div className={`${panelClass} p-6`}>
            <p className="text-sm text-slate-400">Medical disclaimer</p>
            <p className="mt-2 text-sm text-slate-300">{result.disclaimer || 'This is an AI-generated health assessment for informational purposes only.'}</p>
          </div>
        </div>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <div className={`${panelClass} p-6`}>
          <h3 className="text-lg font-semibold text-white">Recommendations</h3>
          <div className="mt-4 space-y-6">
            <div>
              <p className="mb-2 text-sm font-medium uppercase tracking-[0.16em] text-slate-300">Pharmaceutical</p>
              {result.recommendations?.pharmaceutical?.map((item) => (
                <div key={item.medication} className="mb-3 rounded-2xl border border-white/10 bg-white/3 p-4">
                  <p className="font-semibold text-white">{item.medication}</p>
                  <p className="mt-1 text-sm text-slate-300">{item.usage}</p>
                  <p className="mt-2 text-xs text-slate-400">{item.precautions}</p>
                </div>
              ))}
            </div>

            <div>
              <p className="mb-2 text-sm font-medium uppercase tracking-[0.16em] text-slate-300">Herbal</p>
              {result.recommendations?.herbal?.map((item) => (
                <div key={item.remedy} className="mb-3 rounded-2xl border border-white/10 bg-white/3 p-4">
                  <p className="font-semibold text-white">{item.remedy}</p>
                  <p className="mt-1 text-sm text-slate-300">{item.usage}</p>
                  <p className="mt-2 text-xs text-slate-400">{item.precautions}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className={`${panelClass} p-6`}>
          <h3 className="text-lg font-semibold text-white">Medical information</h3>
          <div className="mt-4 space-y-3 text-sm text-slate-300">
            {result.recommendations?.contraindication_warnings?.length ? (
              result.recommendations.contraindication_warnings.map((warning) => (
                <div key={warning} className="flex items-start gap-2 rounded-2xl border border-amber-400/30 bg-amber-500/10 p-3 text-amber-100">
                  <TriangleAlert className="mt-0.5 h-4 w-4" />
                  <span>{warning}</span>
                </div>
              ))
            ) : (
              <p>Medical information is not available for this assessment.</p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

function RiskPage() {
  const result = useAssessmentResult();

  if (!result) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
        <ErrorState message="No assessment result is available yet. Please complete the assessment first." />
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
      <div className={`${panelClass} p-6`}>
        <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-300">Risk</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-white">Current health risk level</h1>
        <div className="mt-6 rounded-[28px] border border-white/10 bg-[#0b0b0b] p-6 text-white">
          <p className="text-sm uppercase tracking-[0.2em] text-slate-300">Risk level</p>
          <div className="mt-3 flex items-center gap-3">
            <BellRing className="h-6 w-6 text-slate-200" />
            <span className="text-3xl font-semibold">{result.severity?.level || 'Unknown'}</span>
          </div>
          <p className="mt-3 text-slate-300">Backend score: {result.severity?.score ?? 0}/100</p>
        </div>
        <div className="mt-6">
          <RiskAlert emergency={result.emergency} />
        </div>
      </div>
    </main>
  );
}

function SHAPPage() {
  const result = useAssessmentResult();

  if (!result) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
        <ErrorState message="No SHAP data is available until an assessment is completed." />
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
      <div className={`${panelClass} p-6`}>
        <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-300">SHAP</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-white">Feature importance</h1>
        <div className="mt-6">
          <SHAPExplanation items={result.shap_importance} />
        </div>
      </div>
    </main>
  );
}

function MedicalInfoPage() {
  const result = useAssessmentResult();

  return (
    <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
      <div className={`${panelClass} p-6`}>
        <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-300">Medical info</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-white">Backend-provided guidance</h1>
        <div className="mt-6 space-y-4">
          {result?.recommendations?.contraindication_warnings?.length ? (
            result.recommendations.contraindication_warnings.map((warning) => (
              <div key={warning} className="rounded-2xl border border-white/10 bg-white/4 p-4 text-sm text-slate-300">
                {warning}
              </div>
            ))
          ) : (
            <div className="rounded-2xl border border-white/10 bg-white/4 p-4 text-sm text-slate-300">
              No additional medical information was provided by the backend for this assessment.
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

function App() {
  const [apiStatus, setApiStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const fetchStatus = async () => {
      try {
        const { data } = await getHealthStatus();
        if (mounted) setApiStatus(data);
      } catch (error) {
        if (mounted) setApiStatus({ status: 'offline', error: 'Backend unavailable' });
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchStatus();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <BrowserRouter>
      <Layout>
        <div className="mx-auto max-w-6xl px-4 py-4 sm:px-6 lg:px-8">
          {!loading && apiStatus && (
            <div className="mb-4 flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200 shadow-[0_0_18px_rgba(255,255,255,0.04)]">
              <span>API status: {apiStatus.status === 'healthy' ? 'Connected' : 'Offline'}</span>
              {apiStatus.components?.database?.status && <span>{apiStatus.components.database.status}</span>}
            </div>
          )}
        </div>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/assessment" element={<AssessmentPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/risk" element={<RiskPage />} />
          <Route path="/shap" element={<SHAPPage />} />
          <Route path="/medical-info" element={<MedicalInfoPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
