import { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  BrainCircuit,
  CheckCircle2,
  Database,
  HeartPulse,
  History,
  Menu,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  User,
  X,
  Zap,
} from 'lucide-react';
import { BrowserRouter, Link, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { assessHealth, getHealthStatus, getHistory } from './lib/api';
import {
  defaultSymptomOptions,
  symptomCategories,
  commonLifestyleFactors,
  commonConditions,
  samplePresets,
} from './data/symptomOptions';
import ErrorState from './components/ErrorState';
import LoadingState from './components/LoadingState';
import RiskAlert from './components/RiskAlert';
import SHAPExplanation from './components/SHAPExplanation';

const initialForm = {
  symptoms: ['fever', 'cough'],
  symptom_text: 'I have a high fever, dry cough, and fatigue for the past 2 days.',
  age: 32,
  lifestyle_factors: ['stress'],
  existing_conditions: [],
};

// ══════════════════════════════════════════════════════════════
//  INTERACTIVE 3D TILT CONTAINER COMPONENT
// ══════════════════════════════════════════════════════════════

function TiltCard({ children, className = '', maxTilt = 8, glare = true, style = {} }) {
  const [tiltStyle, setTiltStyle] = useState('perspective(1000px) rotateX(0deg) rotateY(0deg)');
  const [glarePos, setGlarePos] = useState({ x: 50, y: 50, opacity: 0 });

  const handleMouseMove = (e) => {
    // Only enable 3D tilt on desktop (screens wider than 768px)
    if (window.innerWidth < 768) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = ((y - centerY) / centerY) * -maxTilt;
    const rotateY = ((x - centerX) / centerX) * maxTilt;

    setTiltStyle(
      `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateZ(8px)`
    );
    setGlarePos({ x: (x / rect.width) * 100, y: (y / rect.height) * 100, opacity: 0.12 });
  };

  const handleMouseLeave = () => {
    setTiltStyle('perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(0px)');
    setGlarePos((prev) => ({ ...prev, opacity: 0 }));
  };

  return (
    <div
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        transform: tiltStyle,
        transformStyle: 'preserve-3d',
        transition: 'transform 0.15s ease-out',
        ...style,
      }}
      className={`relative ${className}`}
    >
      {glare && (
        <div
          className="pointer-events-none absolute inset-0 z-30 transition-opacity duration-300 rounded-3xl hidden md:block"
          style={{
            background: `radial-gradient(circle at ${glarePos.x}% ${glarePos.y}%, rgba(255,255,255,${glarePos.opacity}) 0%, transparent 60%)`,
          }}
        />
      )}
      {children}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
//  LAYOUT & RESPONSIVE NAVIGATION
// ══════════════════════════════════════════════════════════════

function Layout({ children, apiStatus }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  const isConnected = apiStatus?.status === 'healthy';
  const dbStatus = apiStatus?.components?.database?.status;

  const navLinks = [
    { path: '/', label: 'Overview' },
    { path: '/assessment', label: 'Diagnostic Studio' },
    { path: '/history', label: 'History' },
    { path: '/medical-info', label: 'Architecture & 50 Diseases' },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#000000] text-zinc-100 antialiased selection:bg-white selection:text-black overflow-x-hidden w-full">
      {/* Minimalist Top Header */}
      <header className="sticky top-0 z-50 border-b border-white/10 bg-[#000000]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-black font-bold shadow-[0_4px_12px_rgba(255,255,255,0.2)] group-hover:scale-105 transition-transform">
              <HeartPulse className="h-4 w-4 stroke-[2.5]" />
            </div>
            <div className="font-display font-bold text-base tracking-tight text-white flex items-center gap-2">
              CureBlend
              <span className="rounded-full bg-zinc-900 border border-zinc-700 px-2 py-0.5 text-[10px] font-mono text-zinc-300">
                AI Health
              </span>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-6 text-sm text-zinc-400">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.path;
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`transition-colors duration-150 py-1 ${
                    isActive ? 'text-white font-semibold' : 'hover:text-zinc-200'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* Actions & Live Status */}
          <div className="hidden sm:flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-[#09090b] px-3 py-1 text-xs text-zinc-400 font-mono shadow-inner">
              <span className={`h-1.5 w-1.5 rounded-full ${isConnected ? 'bg-white' : 'bg-zinc-600'}`} />
              <span>{dbStatus === 'connected' ? 'MongoDB' : isConnected ? 'Local DB' : 'Offline'}</span>
            </div>

            <Link to="/assessment" className="btn-primary text-xs py-2 px-4 shadow-[0_4px_14px_rgba(255,255,255,0.2)]">
              Get Started
            </Link>
          </div>

          {/* Mobile Toggle Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-[#09090b] text-zinc-300"
            aria-label="Toggle Navigation Menu"
          >
            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>

        {/* Mobile Dropdown Menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-b border-white/10 bg-[#09090b] px-4 py-4 space-y-2"
            >
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`block px-3 py-2.5 rounded-xl text-sm font-medium transition ${
                    location.pathname === link.path
                      ? 'bg-white/15 text-white font-semibold'
                      : 'text-zinc-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {link.label}
                </Link>
              ))}

              <div className="pt-3 border-t border-white/10 flex items-center justify-between">
                <span className="text-xs text-zinc-500 font-mono">
                  {dbStatus === 'connected' ? '● MongoDB Connected' : '● Local Store Active'}
                </span>
                <Link
                  to="/assessment"
                  onClick={() => setMobileMenuOpen(false)}
                  className="rounded-full bg-white px-4 py-2 text-xs font-bold text-black"
                >
                  Diagnose
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* Main Content */}
      <div className="flex-1 w-full">{children}</div>

      {/* Minimalist Monochrome Footer */}
      <footer className="border-t border-white/10 bg-[#000000] py-8 text-xs text-zinc-500 mt-16 w-full">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 font-mono text-zinc-400">
            <ShieldCheck className="h-4 w-4 text-white" />
            <span>CureBlend • AI-Powered Personalized Healthcare Assistant</span>
          </div>
          <p className="text-center sm:text-right text-[11px] text-zinc-600 max-w-md">
            Clinical decision support demonstration platform. Always consult a certified healthcare professional.
          </p>
        </div>
      </footer>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
//  HOME / LANDING (RESPONSIVE & TACTILE 3D)
// ══════════════════════════════════════════════════════════════

function LandingPage() {
  const navigate = useNavigate();

  const handlePresetSelect = (preset) => {
    navigate('/assessment', { state: { preset } });
  };

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10 sm:py-16 lg:py-20 space-y-16 sm:space-y-24 perspective-2000 w-full">
      {/* Hero Section */}
      <section className="text-center max-w-4xl mx-auto space-y-6">
        {/* Top Tag Pill */}
        <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3.5 py-1.5 text-xs font-medium text-zinc-300 shadow-[0_4px_20px_rgba(0,0,0,0.5)]">
          <Sparkles className="h-3.5 w-3.5 text-white" />
          <span>Precision Clinical Decision Support & Dual Recommendations</span>
          <ArrowRight className="h-3 w-3 text-zinc-400" />
        </div>

        {/* Big Bold Clean Headline */}
        <h1 className="font-display text-3xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight text-white leading-[1.1]">
          AI-Powered Symptom Checker &{' '}
          <span className="text-zinc-400">Personalized Healthcare Assistant</span>
        </h1>

        {/* Subtitle */}
        <p className="text-sm sm:text-base lg:text-lg text-zinc-400 max-w-2xl mx-auto leading-relaxed px-2">
          Synthesizing natural language symptoms, calibrated multi-class XGBoost models, SHAP explainability, and dual pharmaceutical & herbal remedies across 50 conditions.
        </p>

        {/* Action Buttons */}
        <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3 w-full sm:w-auto px-4">
          <Link to="/assessment" className="btn-primary btn-3d w-full sm:w-auto py-3 px-6 text-sm">
            Get Started Free <ArrowRight className="h-4 w-4" />
          </Link>
          <Link to="/medical-info" className="btn-secondary w-full sm:w-auto py-3 px-6 text-sm">
            View Architecture & 50 Diseases
          </Link>
        </div>
      </section>

      {/* 3D Interactive Hero Dashboard Preview */}
      <section className="max-w-6xl mx-auto w-full">
        <TiltCard maxTilt={5} className="hero-3d-panel rounded-3xl bg-[#09090b] p-4 sm:p-6 lg:p-8 border border-white/10">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-5 border-b border-white/10">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/5 border border-white/10 text-white font-bold">
                <Activity className="h-4 w-4" />
              </div>
              <div>
                <h2 className="text-base font-bold text-white">Diagnostic Dashboard</h2>
                <p className="text-xs text-zinc-400">Interactive Health Decision Engine</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-white/10 border border-white/10 px-3 py-1 text-xs text-zinc-300 font-mono shadow-sm">
                50+ Conditions Supported
              </span>
            </div>
          </div>

          {/* Metric Cards Row */}
          <div className="grid gap-3 sm:gap-4 mt-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-[#121215] p-4 sm:p-5 card-3d">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Model Accuracy</span>
                <span className="text-[10px] font-mono text-zinc-300">XGBoost + Calibrated</span>
              </div>
              <div className="mt-2 sm:mt-3 text-2xl sm:text-3xl font-extrabold text-white font-mono">99.4%</div>
              <div className="mt-1 sm:mt-2 text-xs text-zinc-500">50 calibrated condition classes</div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-[#121215] p-4 sm:p-5 card-3d">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Safety Gate</span>
                <span className="text-[10px] font-mono text-zinc-300">Deterministic</span>
              </div>
              <div className="mt-2 sm:mt-3 text-2xl sm:text-3xl font-extrabold text-white font-mono">100%</div>
              <div className="mt-1 sm:mt-2 text-xs text-zinc-500">Red-flag emergency scan</div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-[#121215] p-4 sm:p-5 card-3d">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Recommendation Modes</span>
                <span className="text-[10px] font-mono text-zinc-300">Dual</span>
              </div>
              <div className="mt-2 sm:mt-3 text-xl sm:text-2xl font-extrabold text-white font-mono">Pharma + Herbal</div>
              <div className="mt-1 sm:mt-2 text-xs text-zinc-500">Filtered for contraindications</div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-[#121215] p-4 sm:p-5 card-3d">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Data Persistence</span>
                <span className="text-[10px] font-mono text-zinc-300">Hybrid</span>
              </div>
              <div className="mt-2 sm:mt-3 text-2xl sm:text-3xl font-extrabold text-white font-mono">MongoDB</div>
              <div className="mt-1 sm:mt-2 text-xs text-zinc-500">With local SQLite fallback</div>
            </div>
          </div>
        </TiltCard>
      </section>

      {/* 1-Click Interactive Case Simulations */}
      <section className="max-w-6xl mx-auto space-y-6 w-full">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-white">Interactive Case Simulations</h2>
            <p className="text-xs text-zinc-400 mt-1">Select a clinical preset to test the decision engine with 1 click</p>
          </div>
        </div>

        <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          {samplePresets.map((preset) => (
            <TiltCard
              key={preset.name}
              maxTilt={6}
              className="card-3d rounded-2xl border border-white/10 bg-[#09090b]"
            >
              <button
                onClick={() => handlePresetSelect(preset)}
                className="w-full text-left p-4 sm:p-5 group cursor-pointer"
              >
                <div className="flex items-center justify-between mb-2 sm:mb-3">
                  <span className="text-[10px] font-mono uppercase font-bold text-zinc-400 border border-white/10 px-2 py-0.5 rounded-full bg-white/5">
                    {preset.badge}
                  </span>
                  <ArrowUpRight className="h-4 w-4 text-zinc-500 group-hover:text-white transition-colors" />
                </div>

                <h3 className="text-sm font-bold text-white group-hover:text-zinc-200">
                  {preset.name}
                </h3>
                <p className="text-xs text-zinc-400 mt-1 line-clamp-2 leading-relaxed">
                  {preset.text}
                </p>
              </button>
            </TiltCard>
          ))}
        </div>
      </section>
    </main>
  );
}

// ══════════════════════════════════════════════════════════════
//  ASSESSMENT STUDIO (FULLY RESPONSIVE)
// ══════════════════════════════════════════════════════════════

function AssessmentPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const presetData = location.state?.preset;

  const [form, setForm] = useState(
    presetData
      ? {
          symptoms: presetData.symptoms || [],
          symptom_text: presetData.text || '',
          age: presetData.age || 30,
          lifestyle_factors: presetData.lifestyle || [],
          existing_conditions: presetData.conditions || [],
        }
      : initialForm
  );

  const [selectedCategory, setSelectedCategory] = useState('Common & General');
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const toggleSymptom = (symptom) => {
    const exists = form.symptoms.includes(symptom);
    const next = exists ? form.symptoms.filter((s) => s !== symptom) : [...form.symptoms, symptom];
    setForm((curr) => ({ ...curr, symptoms: next }));
  };

  const toggleGroupItem = (group, item) => {
    const list = form[group] || [];
    const next = list.includes(item) ? list.filter((i) => i !== item) : [...list, item];
    setForm((curr) => ({ ...curr, [group]: next }));
  };

  const filteredSymptoms = useMemo(() => {
    if (searchTerm.trim()) {
      return defaultSymptomOptions.filter((s) => s.toLowerCase().includes(searchTerm.toLowerCase()));
    }
    return symptomCategories[selectedCategory] || [];
  }, [searchTerm, selectedCategory]);

  const handleSubmit = async (e) => {
    e?.preventDefault?.();
    setError('');

    if (form.symptoms.length === 0 && !form.symptom_text.trim()) {
      setError('Please select symptom chips or provide a symptom narrative.');
      return;
    }

    setIsLoading(true);

    try {
      const payload = {
        symptoms: form.symptoms,
        symptom_text: form.symptom_text.trim(),
        age: Number(form.age) || 30,
        lifestyle_factors: form.lifestyle_factors,
        existing_conditions: form.existing_conditions,
      };

      const { data } = await assessHealth(payload);
      navigate('/results', { state: { result: data } });
    } catch (err) {
      console.error('Assessment error:', err);
      setError(err?.response?.data?.detail || err?.message || 'Failed to complete assessment.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-6 sm:space-y-8 w-full">
      <div>
        <h1 className="font-display text-2xl sm:text-3xl font-extrabold text-white">Diagnostic Studio</h1>
        <p className="text-xs text-zinc-400 mt-1">Select patient symptoms, demographics, and clinical context</p>
      </div>

      <form onSubmit={handleSubmit} className="grid gap-6 lg:gap-8 grid-cols-1 lg:grid-cols-12 w-full">
        {/* Left Column: Symptoms (7 cols on lg) */}
        <div className="space-y-6 lg:col-span-7">
          {/* Symptom Selector */}
          <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <h2 className="text-sm font-bold text-white">
                Symptom Chips ({form.symptoms.length} selected)
              </h2>

              <div className="relative w-full sm:w-48">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Filter symptoms..."
                  className="w-full rounded-full border border-white/10 bg-[#121215] pl-8 pr-3 py-1.5 text-xs text-zinc-200 placeholder-zinc-500 outline-none focus:border-white/30"
                />
              </div>
            </div>

            {/* Category tabs */}
            {!searchTerm && (
              <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar pb-3 mb-3 border-b border-white/10 whitespace-nowrap">
                {Object.keys(symptomCategories).map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => setSelectedCategory(cat)}
                    className={`rounded-full px-3 py-1.5 text-xs font-medium transition cursor-pointer shrink-0 ${
                      selectedCategory === cat
                        ? 'bg-white text-black font-semibold'
                        : 'bg-white/5 text-zinc-400 hover:text-white'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            )}

            {/* Chips Grid */}
            <div className="flex flex-wrap gap-1.5 sm:gap-2 max-h-[240px] overflow-y-auto pr-1">
              {filteredSymptoms.map((symptom) => {
                const isSelected = form.symptoms.includes(symptom);
                return (
                  <button
                    key={symptom}
                    type="button"
                    onClick={() => toggleSymptom(symptom)}
                    className={`rounded-lg px-2.5 sm:px-3 py-1.5 text-xs transition flex items-center gap-1.5 cursor-pointer ${
                      isSelected
                        ? 'bg-white text-black font-bold shadow-sm'
                        : 'bg-[#121215] text-zinc-300 border border-white/10 hover:border-white/20'
                    }`}
                  >
                    {isSelected && <CheckCircle2 className="h-3 w-3" />}
                    {symptom}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Symptom Narrative */}
          <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6">
            <h2 className="text-sm font-bold text-white mb-1">Clinical Narrative (NLP)</h2>
            <p className="text-xs text-zinc-400 mb-3">Describe patient symptoms in natural conversational language</p>
            <textarea
              rows={4}
              value={form.symptom_text}
              onChange={(e) => setForm({ ...form, symptom_text: e.target.value })}
              placeholder="e.g. High fever for 2 days with dry cough, chills, and severe body aches..."
              className="w-full rounded-xl border border-white/10 bg-[#121215] p-3.5 text-xs text-zinc-200 placeholder-zinc-500 outline-none focus:border-white/30 resize-none leading-relaxed"
            />
          </div>
        </div>

        {/* Right Column: Demographics & Submit (5 cols on lg) */}
        <div className="space-y-6 lg:col-span-5">
          {/* Patient Age Slider */}
          <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-bold text-white">Patient Age</h2>
              <span className="text-xs font-mono font-bold text-white bg-white/10 px-2.5 py-0.5 rounded-md border border-white/10">
                {form.age} yrs
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              value={form.age}
              onChange={(e) => setForm({ ...form, age: Number(e.target.value) })}
              className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-white"
            />
          </div>

          {/* Pre-Existing Conditions */}
          <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6">
            <h2 className="text-sm font-bold text-white mb-2">Pre-Existing Conditions</h2>
            <div className="flex flex-wrap gap-1.5">
              {commonConditions.map((cond) => {
                const active = form.existing_conditions.includes(cond);
                return (
                  <button
                    key={cond}
                    type="button"
                    onClick={() => toggleGroupItem('existing_conditions', cond)}
                    className={`rounded-lg px-2.5 py-1 text-xs transition cursor-pointer ${
                      active
                        ? 'bg-white text-black font-semibold'
                        : 'bg-[#121215] text-zinc-400 border border-white/5 hover:border-white/20'
                    }`}
                  >
                    {cond}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Lifestyle Factors */}
          <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6">
            <h2 className="text-sm font-bold text-white mb-2">Lifestyle Risk Factors</h2>
            <div className="flex flex-wrap gap-1.5">
              {commonLifestyleFactors.map((factor) => {
                const active = form.lifestyle_factors.includes(factor);
                return (
                  <button
                    key={factor}
                    type="button"
                    onClick={() => toggleGroupItem('lifestyle_factors', factor)}
                    className={`rounded-lg px-2.5 py-1 text-xs transition cursor-pointer ${
                      active
                        ? 'bg-white text-black font-semibold'
                        : 'bg-[#121215] text-zinc-400 border border-white/5 hover:border-white/20'
                    }`}
                  >
                    {factor}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full btn-primary btn-3d py-3.5 text-sm font-bold disabled:opacity-50 cursor-pointer"
          >
            {isLoading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            {isLoading ? 'Analyzing Clinical Signals...' : 'Run Assessment'}
          </button>
        </div>
      </form>

      {isLoading && <LoadingState />}
      {error && <ErrorState message={error} onRetry={handleSubmit} />}
    </main>
  );
}

// ══════════════════════════════════════════════════════════════
//  RESULTS DASHBOARD (FULLY RESPONSIVE)
// ══════════════════════════════════════════════════════════════

function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;
  const [activeTab, setActiveTab] = useState('pharmaceutical');

  if (!result) {
    return (
      <main className="mx-auto max-w-4xl px-4 py-16 text-center">
        <ErrorState message="No assessment record found in current session." />
        <button onClick={() => navigate('/assessment')} className="mt-4 btn-primary text-xs">
          Go to Assessment Studio
        </button>
      </main>
    );
  }

  const topPrediction = result.predictions?.[0];
  const severity = result.severity || { level: 'Moderate', score: 50 };
  const emergency = result.emergency || { is_emergency: false };

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-6 sm:space-y-8 w-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-white/10">
        <div>
          <span className="text-xs font-mono text-zinc-500">ID: {result.request_id?.slice(0, 8)}</span>
          <h1 className="font-display text-2xl sm:text-3xl font-extrabold text-white mt-0.5">Clinical Assessment Results</h1>
        </div>
        <div className="flex gap-2">
          <Link to="/assessment" className="btn-secondary text-xs py-2 px-3">
            <RefreshCw className="h-3.5 w-3.5" /> Re-Assess
          </Link>
          <Link to="/history" className="btn-primary text-xs py-2 px-3">
            <History className="h-3.5 w-3.5" /> View History
          </Link>
        </div>
      </div>

      {/* 1. Emergency Banner (If Triggered) */}
      <RiskAlert emergency={emergency} />

      {/* 2. 3D Diagnosis & Severity Cards Grid */}
      <div className="grid gap-4 sm:gap-6 grid-cols-1 md:grid-cols-3 w-full">
        {/* Top Match (2 cols on md) */}
        <TiltCard maxTilt={5} className="md:col-span-2 rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6 card-3d">
          <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2">
            <div>
              <span className="text-xs font-mono uppercase text-zinc-500">Top Predicted Diagnosis</span>
              <h2 className="text-2xl sm:text-3xl font-bold text-white mt-1">{topPrediction?.condition || 'Unknown'}</h2>
            </div>
            <div className="text-left sm:text-right">
              <span className="text-2xl sm:text-3xl font-extrabold font-mono text-white">
                {((topPrediction?.probability || 0) * 100).toFixed(1)}%
              </span>
              <p className="text-[10px] font-mono text-zinc-500">Confidence Probability</p>
            </div>
          </div>

          <div className="mt-4 h-2.5 w-full bg-[#18181b] rounded-full overflow-hidden">
            <div
              className="h-full bg-white rounded-full transition-all duration-700 shadow-[0_0_10px_rgba(255,255,255,0.4)]"
              style={{ width: `${(topPrediction?.probability || 0) * 100}%` }}
            />
          </div>

          {/* Differential Top-K List */}
          <div className="mt-6 pt-4 border-t border-white/10 space-y-2">
            <span className="text-[11px] font-mono text-zinc-500 uppercase">Differential Probabilities</span>
            {result.predictions?.map((pred) => (
              <div key={pred.condition} className="flex items-center justify-between text-xs py-0.5">
                <span className="text-zinc-300 font-medium">{pred.condition}</span>
                <span className="font-mono text-zinc-400 font-bold">{(pred.probability * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </TiltCard>

        {/* Severity Score (1 col on md) */}
        <TiltCard maxTilt={5} className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6 flex flex-col justify-between card-3d">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono uppercase text-zinc-500">Severity Level</span>
              <span className="text-xs font-bold font-mono px-2.5 py-0.5 rounded-full border border-white/20 bg-white/5 text-white">
                {severity.level}
              </span>
            </div>

            <div className="flex items-baseline gap-2 mt-4">
              <span className="text-4xl sm:text-5xl font-extrabold font-mono text-white">{severity.score}</span>
              <span className="text-zinc-500 font-mono text-sm">/ 100</span>
            </div>

            <div className="mt-4 h-2 w-full bg-[#18181b] rounded-full overflow-hidden">
              <div className="h-full bg-white rounded-full" style={{ width: `${severity.score}%` }} />
            </div>
          </div>

          {severity.breakdown && (
            <div className="mt-6 pt-4 border-t border-white/10 text-[11px] font-mono text-zinc-500 space-y-1">
              <div className="flex justify-between">
                <span>Disease Base:</span>
                <span className="text-zinc-300">{severity.breakdown.disease_base_weight ?? '-'}</span>
              </div>
              <div className="flex justify-between">
                <span>Symptom Load:</span>
                <span className="text-zinc-300">+{severity.breakdown.symptom_load ?? '-'}</span>
              </div>
              <div className="flex justify-between">
                <span>Age Risk Factor:</span>
                <span className="text-zinc-300">+{severity.breakdown.age_factor ?? '-'}</span>
              </div>
            </div>
          )}
        </TiltCard>
      </div>

      {/* 3. SHAP Explainability Component */}
      <SHAPExplanation items={result.shap_importance} />

      {/* 4. Dual Recommendations */}
      <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-white/10">
          <div>
            <h2 className="text-base font-bold text-white">Evidence-Based Dual Recommendations</h2>
            <p className="text-xs text-zinc-400">Targeted clinical guidance for {topPrediction?.condition}</p>
          </div>

          <div className="flex items-center gap-1 bg-[#121215] p-1 rounded-full border border-white/10 shrink-0">
            <button
              onClick={() => setActiveTab('pharmaceutical')}
              className={`rounded-full px-3.5 sm:px-4 py-1.5 text-xs font-semibold transition cursor-pointer ${
                activeTab === 'pharmaceutical' ? 'bg-white text-black' : 'text-zinc-400 hover:text-white'
              }`}
            >
              Pharmaceutical ({result.recommendations?.pharmaceutical?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('herbal')}
              className={`rounded-full px-3.5 sm:px-4 py-1.5 text-xs font-semibold transition cursor-pointer ${
                activeTab === 'herbal' ? 'bg-white text-black' : 'text-zinc-400 hover:text-white'
              }`}
            >
              Herbal & Lifestyle ({result.recommendations?.herbal?.length || 0})
            </button>
          </div>
        </div>

        {/* Contraindications Warnings */}
        {result.recommendations?.contraindication_warnings?.length > 0 && (
          <div className="mb-6 space-y-2">
            {result.recommendations.contraindication_warnings.map((warn, idx) => (
              <div key={idx} className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200 flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                <span>{warn}</span>
              </div>
            ))}
          </div>
        )}

        {/* Active Tab Items */}
        <div className="grid gap-3 sm:gap-4 grid-cols-1 md:grid-cols-2">
          {activeTab === 'pharmaceutical' &&
            result.recommendations?.pharmaceutical?.map((med, idx) => (
              <div key={idx} className="rounded-xl border border-white/10 bg-[#121215] p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-bold text-white">{med.medication}</h3>
                  <span className="text-[10px] font-mono bg-white/10 text-zinc-300 px-2 py-0.5 rounded">
                    Pharma
                  </span>
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">{med.usage}</p>
                {med.precautions && (
                  <p className="mt-2 text-[11px] text-zinc-500 italic">{med.precautions}</p>
                )}
              </div>
            ))}

          {activeTab === 'herbal' &&
            result.recommendations?.herbal?.map((herb, idx) => (
              <div key={idx} className="rounded-xl border border-white/10 bg-[#121215] p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-bold text-white">{herb.remedy}</h3>
                  <span className="text-[10px] font-mono bg-white/10 text-zinc-300 px-2 py-0.5 rounded">
                    Herbal
                  </span>
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">{herb.usage}</p>
                {herb.precautions && (
                  <p className="mt-2 text-[11px] text-zinc-500 italic">{herb.precautions}</p>
                )}
              </div>
            ))}
        </div>
      </div>
    </main>
  );
}

// ══════════════════════════════════════════════════════════════
//  HISTORY TIMELINE (FULLY RESPONSIVE)
// ══════════════════════════════════════════════════════════════

function HistoryPage() {
  const [historyItems, setHistoryItems] = useState([]);
  const [totalRecords, setTotalRecords] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const { data } = await getHistory(50, 0);
      setHistoryItems(data.records || []);
      setTotalRecords(data.total_records || 0);
    } catch (err) {
      console.error('History error:', err);
      setError('Unable to fetch history from database.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-6 w-full">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-white/10">
        <div>
          <h1 className="font-display text-2xl sm:text-3xl font-extrabold text-white">Assessment History</h1>
          <p className="text-xs text-zinc-400 mt-1">Past patient records logged to MongoDB / Local Store</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-zinc-400 bg-[#09090b] px-3 py-1.5 rounded-full border border-white/10">
            Total: {totalRecords}
          </span>
          <button onClick={fetchHistory} className="btn-secondary text-xs py-1.5 px-3 cursor-pointer">
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>
      </div>

      {loading ? (
        <LoadingState message="Fetching database records..." />
      ) : error ? (
        <ErrorState message={error} onRetry={fetchHistory} />
      ) : historyItems.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-[#09090b] p-8 sm:p-12 text-center text-zinc-400">
          <History className="h-8 w-8 mx-auto mb-2 text-zinc-600" />
          <h3 className="text-base font-bold text-white">No Records Yet</h3>
          <Link to="/assessment" className="mt-4 inline-flex btn-primary text-xs">
            Start New Assessment
          </Link>
        </div>
      ) : (
        <div className="space-y-3">
          {historyItems.map((item) => (
            <div
              key={item.request_id}
              className="rounded-xl border border-white/10 bg-[#09090b] p-4 flex flex-col md:flex-row md:items-center justify-between gap-3"
            >
              <div>
                <div className="flex items-center gap-2 text-xs text-zinc-500 font-mono">
                  <span>ID: {item.request_id?.slice(0, 8)}</span>
                  <span>•</span>
                  <span>{new Date(item.timestamp).toLocaleString()}</span>
                  {item.is_emergency && (
                    <span className="border border-red-500/40 bg-red-500/10 text-red-400 px-1.5 py-0.2 rounded text-[10px]">
                      EMERGENCY
                    </span>
                  )}
                </div>
                <div className="flex items-baseline gap-2 mt-1">
                  <h3 className="text-base font-bold text-white">{item.top_prediction}</h3>
                  <span className="text-xs font-mono text-zinc-400">
                    {(item.confidence * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-zinc-500">| Severity: {item.severity_level}</span>
                </div>
              </div>

              {item.input_symptoms?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {item.input_symptoms.map((s) => (
                    <span key={s} className="rounded bg-[#18181b] px-2 py-0.5 text-[10px] text-zinc-300 font-mono">
                      {s}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </main>
  );
}

// ══════════════════════════════════════════════════════════════
//  TECHNICAL ARCHITECTURE & 50 DISEASES (FULLY RESPONSIVE)
// ══════════════════════════════════════════════════════════════

function MedicalInfoPage() {
  const [search, setSearch] = useState('');

  const diseases = [
    { name: "Influenza (Flu)", symptoms: "high_fever, body_ache, chills, dry_cough, fatigue" },
    { name: "Common Cold", symptoms: "sneezing, runny_nose, nasal_congestion, sore_throat" },
    { name: "COVID-19", symptoms: "loss_of_taste, loss_of_smell, fever, dry_cough" },
    { name: "Pneumonia", symptoms: "productive_cough, yellow_sputum, high_fever, pleuritic chest pain" },
    { name: "Bronchial Asthma", symptoms: "wheezing, shortness_of_breath, chest_tightness, nocturnal_cough" },
    { name: "COPD", symptoms: "chronic_cough, productive_cough, breathlessness on exertion" },
    { name: "Tuberculosis", symptoms: "chronic_cough (>3 wks), blood_in_sputum, night_sweats, weight_loss" },
    { name: "Dengue Fever", symptoms: "high_fever, retro_orbital_pain, severe_joint_pain, skin_rash" },
    { name: "Malaria", symptoms: "cyclical_fever, shivering_chills, profuse_sweating, vomiting" },
    { name: "Typhoid Fever", symptoms: "step_ladder_fever, abdominal_pain, headache, rose_spots" },
    { name: "Gastroenteritis", symptoms: "watery_diarrhea, vomiting, abdominal_cramps, dehydration" },
    { name: "GERD (Acid Reflux)", symptoms: "heartburn, acid_reflux, sour_taste, chest_burning" },
    { name: "Peptic Ulcer Disease", symptoms: "epigastric_pain on empty stomach, burning stomach ache" },
    { name: "Appendicitis", symptoms: "right_lower_quadrant_pain, periumbilical pain, fever, nausea" },
    { name: "Gallstone", symptoms: "right_upper_quadrant_pain radiating to shoulder, nausea after fatty meal" },
    { name: "Diverticulitis", symptoms: "left_lower_quadrant_pain, fever, constipation, tenderness" },
    { name: "Kidney Stones", symptoms: "severe_flank_pain radiating to groin, blood in urine" },
    { name: "Urinary Tract Infection", symptoms: "burning_urination, dysuria, frequent urge, pelvic pain" },
    { name: "Migraine", symptoms: "throbbing_headache, unilateral_headache, photophobia, visual aura" },
    { name: "Hypertension", symptoms: "high_blood_pressure, morning_headache, dizziness, blurry vision" },
    { name: "Heart Attack", symptoms: "crushing_chest_pain radiating to left arm/jaw, cold sweat" },
    { name: "Heart Failure", symptoms: "shortness_of_breath lying flat (orthopnea), pedal edema" },
    { name: "Diabetes Type 2", symptoms: "frequent_urination (polyuria), excessive_thirst, hunger" },
    { name: "Hypoglycemia", symptoms: "shakiness, cold_sweat, rapid_heartbeat, confusion, extreme hunger" },
    { name: "Hyperthyroidism", symptoms: "rapid_weight_loss, palpitations, heat_intolerance, tremors" },
    { name: "Hypothyroidism", symptoms: "weight_gain, extreme_fatigue, cold_intolerance, dry skin" },
    { name: "Anemia", symptoms: "extreme_fatigue, pale_skin, dizziness, cold hands, brittle nails" },
    { name: "Hepatitis", symptoms: "jaundice, yellow_eyes, dark_urine, pale stool, liver pain" },
    { name: "Jaundice", symptoms: "yellow skin, scleral icterus, dark tea colored urine, pruritus" },
    { name: "Arthritis", symptoms: "joint_pain, morning_stiffness over 1 hour, joint swelling" },
    { name: "Gout", symptoms: "excruciating big_toe_pain (podagra), joint redness, nocturnal swelling" },
    { name: "Cervical Spondylosis", symptoms: "neck_pain, neck stiffness, radiating arm pain, hand numbness" },
    { name: "Allergic Rhinitis", symptoms: "sneezing bouts, itchy watery eyes, clear runny nose" },
    { name: "Allergy (Urticaria)", symptoms: "skin_rash, red raised hives, generalized itching" },
    { name: "Eczema", symptoms: "dry itchy scaly skin patches in elbow/knee creases, cracking" },
    { name: "Psoriasis", symptoms: "well demarcated red plaques with silvery scales, pitted nails" },
    { name: "Acne", symptoms: "pimples, blackheads, whiteheads, cystic lesions, oily skin" },
    { name: "Fungal Infection", symptoms: "ring_shaped_rash with scaly borders, intense itching" },
    { name: "Chicken Pox", symptoms: "crops of fluid filled itchy blisters (vesicles), fever" },
    { name: "Strep Throat", symptoms: "severe sore throat, painful swallowing, tonsillar exudate, no cough" },
    { name: "Acute Sinusitis", symptoms: "facial pressure under eyes, thick yellow discharge, headache" },
    { name: "Acute Bronchitis", symptoms: "deep hacking cough with yellow mucus, chest burning" },
    { name: "Conjunctivitis", symptoms: "bloodshot pink eyes, yellow crusty morning discharge, gritty eyes" },
    { name: "Hemorrhoids", symptoms: "bright red rectal bleeding on tissue, anal pain and swelling" },
    { name: "Varicose Veins", symptoms: "twisted bulging blue veins, aching heavy legs, ankle swelling" },
    { name: "Sepsis", symptoms: "high fever with violent shivering, tachycardia >120bpm, hypotension" },
    { name: "AIDS / HIV", symptoms: "prolonged fever, chronic diarrhea, night sweats, oral thrush" },
    { name: "Drug Reaction", symptoms: "sudden widespread rash and hives after starting new medication" },
    { name: "Depression", symptoms: "persistent sadness, anhedonia (loss of interest), low energy, insomnia" },
    { name: "Anxiety", symptoms: "panic attacks, racing heart, hyperventilation, constant restlessness" },
  ];

  const filtered = diseases.filter(
    (d) =>
      d.name.toLowerCase().includes(search.toLowerCase()) ||
      d.symptoms.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <main className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-8 w-full">
      <div>
        <h1 className="font-display text-2xl sm:text-3xl font-extrabold text-white">System Architecture & 50 Diseases</h1>
        <p className="text-xs text-zinc-400 mt-1">Technical specifications and master clinical conditions reference</p>
      </div>

      <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2">
        <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-5 space-y-2">
          <div className="flex items-center gap-2 font-bold text-white text-sm">
            <BrainCircuit className="h-4 w-4" /> Multi-Class Classifier (99.4% Accuracy)
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Calibrated XGBoost model trained on multi-hot symptom chips, TF-IDF text features, and scaled demographic variables.
          </p>
        </div>

        <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-5 space-y-2">
          <div className="flex items-center gap-2 font-bold text-white text-sm">
            <Activity className="h-4 w-4" /> SHAP Feature Attribution
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            TreeExplainer mathematically calculates exact positive and negative contribution weights for individual symptoms.
          </p>
        </div>

        <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-5 space-y-2">
          <div className="flex items-center gap-2 font-bold text-white text-sm">
            <ShieldCheck className="h-4 w-4" /> Deterministic Red-Flag Gate
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Emergency keyword scan triggers acute triage warnings prior to returning standard recommendations.
          </p>
        </div>

        <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-5 space-y-2">
          <div className="flex items-center gap-2 font-bold text-white text-sm">
            <Database className="h-4 w-4" /> Hybrid Storage Engine
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Async Motor driver connects directly to MongoDB with auto-recovering local SQLite fallback.
          </p>
        </div>
      </div>

      {/* 50 Diseases Reference Directory */}
      <div className="rounded-2xl border border-white/10 bg-[#09090b] p-4 sm:p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
          <div>
            <h2 className="text-base font-bold text-white">50 Supported Clinical Conditions</h2>
            <p className="text-xs text-zinc-400">Search for any condition or symptom below</p>
          </div>

          <div className="relative w-full sm:w-60">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search diseases or symptoms..."
              className="w-full rounded-full border border-white/10 bg-[#121215] pl-8 pr-3 py-1.5 text-xs text-zinc-200 placeholder-zinc-500 outline-none focus:border-white/30"
            />
          </div>
        </div>

        <div className="grid gap-2 sm:grid-cols-2 max-h-[400px] overflow-y-auto pr-1">
          {filtered.map((d, i) => (
            <div key={d.name} className="rounded-xl border border-white/5 bg-[#121215] p-3">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-white">{i + 1}. {d.name}</span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-1 font-mono leading-relaxed">
                {d.symptoms}
              </p>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}

// ══════════════════════════════════════════════════════════════
//  APP ROOT
// ══════════════════════════════════════════════════════════════

export default function App() {
  const [apiStatus, setApiStatus] = useState(null);

  useEffect(() => {
    let mounted = true;
    const checkStatus = async () => {
      try {
        const { data } = await getHealthStatus();
        if (mounted) setApiStatus(data);
      } catch (err) {
        if (mounted) setApiStatus({ status: 'offline', error: 'Backend unreachable' });
      }
    };
    checkStatus();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <BrowserRouter>
      <Layout apiStatus={apiStatus}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/assessment" element={<AssessmentPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/medical-info" element={<MedicalInfoPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
