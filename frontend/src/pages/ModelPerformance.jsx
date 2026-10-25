import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line
} from 'recharts'
import { Play } from 'lucide-react'
import { getEvaluation, trainModels } from '../api/client'
import { Card, SectionTitle, Spinner, AlertBanner } from '../components/UI'

function ConfusionMatrix({ cm }) {
  const cells = [
    { label: 'True Negative (Safe)', value: cm.tn, bg: 'rgba(16,185,129,0.06)', color: '#6ee7b7' },
    { label: 'False Positive (Alert)', value: cm.fp, bg: 'rgba(245,158,11,0.06)', color: '#fcd34d' },
    { label: 'False Negative (Missed)', value: cm.fn, bg: 'rgba(244,63,94,0.06)', color: '#fda4af' },
    { label: 'True Positive (Caught)', value: cm.tp, bg: 'rgba(16,185,129,0.06)', color: '#6ee7b7' },
  ]
  return (
    <div>
      <p className="text-xs font-medium text-zinc-400 mb-2">Confusion Matrix</p>
      <div className="grid grid-cols-2 gap-2">
        {cells.map((c, i) => (
          <div key={i} className="rounded-md p-2.5 text-center border border-white/[0.05]" style={{ background: c.bg }}>
            <p className="text-base font-semibold tracking-tight" style={{ color: c.color }}>{c.value.toLocaleString()}</p>
            <p className="text-[10px] text-zinc-400 mt-0.5">{c.label}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function MetricBar({ label, value, color }) {
  const pct = Math.round(value * 100)
  return (
    <div>
      <div className="flex justify-between mb-1 text-xs">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono font-medium text-zinc-200">{pct}%</span>
      </div>
      <div className="h-1 rounded-full bg-[#18191d] overflow-hidden">
        <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

export default function ModelPerformance() {
  const [selectedModel, setSelectedModel] = useState(null)
  const [trainConfig, setTrainConfig] = useState({
    model_names: ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    test_size: 0.2,
    use_smote: true
  })

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['evaluation'],
    queryFn: () => getEvaluation().then(r => r.data),
    retry: false,
  })

  const trainMut = useMutation({
    mutationFn: () => trainModels(trainConfig).then(r => r.data),
    onSuccess: () => { setTimeout(() => refetch(), 500) },
  })

  const activeModel = selectedModel
    ? data?.models?.find(m => m.model_name === selectedModel)
    : data?.models?.find(m => m.model_name === data?.best_model)

  const comparisonData = data?.models?.map(m => ({
    name: m.model_name,
    Accuracy: +(m.accuracy * 100).toFixed(1),
    Precision: +(m.precision * 100).toFixed(1),
    Recall: +(m.recall * 100).toFixed(1),
    'F1-Score': +(m.f1_score * 100).toFixed(1),
  }))

  return (
    <div className="page-enter space-y-4">
      {/* Training Controls */}
      <Card>
        <SectionTitle subtitle="Configure hyperparameters and train on loaded records">
          Model Training
        </SectionTitle>
        <div className="flex flex-wrap gap-4 items-center mb-3">
          <div>
            <p className="text-xs text-zinc-400 mb-1.5">Algorithms</p>
            <div className="flex gap-1.5">
              {['Logistic Regression', 'Decision Tree', 'Random Forest'].map(m => {
                const checked = trainConfig.model_names.includes(m)
                return (
                  <button
                    key={m}
                    onClick={() => setTrainConfig(c => ({
                      ...c,
                      model_names: checked ? c.model_names.filter(x => x !== m) : [...c.model_names, m]
                    }))}
                    className={`text-xs px-2.5 py-1 rounded transition-colors cursor-pointer border ${
                      checked
                        ? 'bg-white/[0.1] text-zinc-100 border-white/[0.2] font-medium'
                        : 'bg-transparent text-zinc-400 border-white/[0.08] hover:text-zinc-200'
                    }`}
                  >
                    {m}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="text-xs text-zinc-400">
            <p className="mb-1">Test Split: {Math.round(trainConfig.test_size * 100)}%</p>
            <input
              type="range"
              min={10} max={40} step={5}
              value={trainConfig.test_size * 100}
              onChange={e => setTrainConfig(c => ({ ...c, test_size: e.target.value / 100 }))}
              className="accent-zinc-200"
            />
          </div>

          <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer pt-3">
            <input
              type="checkbox"
              checked={trainConfig.use_smote}
              onChange={e => setTrainConfig(c => ({ ...c, use_smote: e.target.checked }))}
              className="accent-zinc-200"
            />
            <span>SMOTE (imbalance resampling)</span>
          </label>
        </div>

        <button
          onClick={() => trainMut.mutate()}
          disabled={trainMut.isPending || trainConfig.model_names.length === 0}
          className="btn-primary flex items-center gap-1.5 text-xs"
        >
          {trainMut.isPending ? <><Spinner size={12} /> Training in progress…</> : <><Play size={12} /> Train Models</>}
        </button>

        {trainMut.isError && <AlertBanner type="error" className="mt-2">{trainMut.error.message}</AlertBanner>}
        {trainMut.isSuccess && (
          <AlertBanner type="success" className="mt-2">
            Models trained: {trainMut.data.models_trained.join(', ')} ({trainMut.data.elapsed_seconds}s)
          </AlertBanner>
        )}
      </Card>

      {isLoading && <div className="glass-card p-6 shimmer h-28" />}

      {error && !trainMut.isSuccess && (
        <AlertBanner type="info">
          No trained models found. Train models above or upload a dataset to begin.
        </AlertBanner>
      )}

      {data && (
        <>
          {/* Algorithm tabs */}
          <div className="flex gap-1.5 flex-wrap">
            {data.models.map(m => {
              const active = (selectedModel || data.best_model) === m.model_name
              return (
                <button
                  key={m.model_name}
                  onClick={() => setSelectedModel(m.model_name)}
                  className={`px-3 py-1.5 rounded text-xs transition-colors cursor-pointer border ${
                    active
                      ? 'bg-white/[0.1] text-zinc-100 border-white/[0.2] font-medium'
                      : 'bg-[#121316] text-zinc-400 border-white/[0.06] hover:text-zinc-200'
                  }`}
                >
                  {m.model_name} {m.model_name === data.best_model && '• Best'}
                </button>
              )
            })}
          </div>

          {/* Metric Comparison Chart */}
          <Card>
            <SectionTitle subtitle="Evaluated strictly on held-out test split">
              Metric Comparison
            </SectionTitle>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={comparisonData} barGap={2} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
                <Bar dataKey="Accuracy" fill="#60a5fa" radius={[2, 2, 0, 0]} opacity={0.8} />
                <Bar dataKey="Precision" fill="#34d399" radius={[2, 2, 0, 0]} opacity={0.8} />
                <Bar dataKey="Recall" fill="#f43f5e" radius={[2, 2, 0, 0]} opacity={0.8} />
                <Bar dataKey="F1-Score" fill="#f59e0b" radius={[2, 2, 0, 0]} opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {activeModel && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <Card>
                <SectionTitle subtitle={`Metrics for ${activeModel.model_name}`}>
                  Detailed Scores
                </SectionTitle>
                <div className="space-y-3 mb-4">
                  <MetricBar label="Accuracy" value={activeModel.accuracy} color="#60a5fa" />
                  <MetricBar label="Precision (minimizes false alarms)" value={activeModel.precision} color="#34d399" />
                  <MetricBar label="Recall (captures total fraud)" value={activeModel.recall} color="#f43f5e" />
                  <MetricBar label="F1-Score (harmonic mean)" value={activeModel.f1_score} color="#f59e0b" />
                  {activeModel.roc_auc != null && <MetricBar label="ROC-AUC" value={activeModel.roc_auc} color="#a78bfa" />}
                </div>

                <div className="p-2.5 rounded text-[11px] text-zinc-400 bg-white/[0.02] border border-white/[0.05]">
                  <span className="text-zinc-300 font-medium">Evaluation Note: </span>
                  Due to extreme class imbalance (~0.2% fraud), Recall and Precision provide the true benchmark of detection quality rather than raw accuracy.
                </div>
              </Card>

              <Card>
                <ConfusionMatrix cm={activeModel.confusion_matrix} />
                {activeModel.roc_curve && (
                  <div className="mt-3">
                    <p className="text-xs text-zinc-400 mb-1.5 font-medium">ROC Curve (AUC = {activeModel.roc_auc?.toFixed(3)})</p>
                    <ResponsiveContainer width="100%" height={140}>
                      <LineChart data={activeModel.roc_curve} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                        <XAxis dataKey="fpr" tick={{ fontSize: 9, fill: '#71717a' }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 9, fill: '#71717a' }} axisLine={false} tickLine={false} />
                        <Tooltip formatter={(v) => v.toFixed(3)} />
                        <Line type="monotone" dataKey="tpr" stroke="#60a5fa" strokeWidth={1.5} dot={false} name="ROC" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </Card>
            </div>
          )}

          {/* Performance Table */}
          <Card>
            <SectionTitle subtitle="Test split benchmarks side-by-side">Performance Summary</SectionTitle>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {data.models.map(m => (
                    <tr key={m.model_name}>
                      <td className="font-medium text-zinc-200">
                        {m.model_name} {m.model_name === data.best_model && <span className="text-[10px] text-amber-300 ml-1 font-mono">BEST</span>}
                      </td>
                      <td className="font-mono">{(m.accuracy * 100).toFixed(2)}%</td>
                      <td className="font-mono">{(m.precision * 100).toFixed(2)}%</td>
                      <td className="font-mono text-rose-300">{(m.recall * 100).toFixed(2)}%</td>
                      <td className="font-mono text-amber-300">{(m.f1_score * 100).toFixed(2)}%</td>
                      <td className="font-mono">{m.roc_auc != null ? m.roc_auc.toFixed(4) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
