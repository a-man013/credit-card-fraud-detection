import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell
} from 'recharts'
import { getInsights } from '../api/client'
import { Card, SectionTitle, Badge, AlertBanner, EmptyState } from '../components/UI'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <p className="text-xs font-medium text-zinc-300 mb-1">{label}</p>
      {payload.map(p => (
        <div key={p.name} className="flex items-center gap-2 text-xs">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: p.fill || p.color }} />
          <span className="text-zinc-400">{p.name}:</span>
          <span className="font-medium text-zinc-100">{p.value?.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

export default function FraudInsights() {
  const [minAmount, setMinAmount] = useState(0)
  const [maxAmount, setMaxAmount] = useState(10000)
  const [classFilter, setClassFilter] = useState('all')

  const { data, isLoading, error } = useQuery({
    queryKey: ['insights'],
    queryFn: () => getInsights().then(r => r.data),
  })

  if (isLoading) return <div className="page-enter"><div className="glass-card h-56 shimmer" /></div>
  if (error) return <div className="page-enter"><AlertBanner type="error">{error.message}</AlertBanner></div>

  const filteredDist = (data.amount_distribution || []).filter(b =>
    b.bin_end >= minAmount && b.bin_start <= maxAmount
  )

  const anomalies = (data.anomalies || []).filter(a => {
    if (classFilter === 'fraud') return a.is_fraud === true
    if (classFilter === 'genuine') return a.is_fraud === false
    return true
  })

  const { stats } = data

  return (
    <div className="page-enter space-y-4">
      {data.is_sample_data && (
        <AlertBanner type="warning">
          Sample dataset insights shown. Upload production records for empirical variance.
        </AlertBanner>
      )}

      {/* Key Metric Overview */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Total Volume', value: stats.total_rows?.toLocaleString() },
          { label: 'Average Amount', value: stats.avg_amount != null ? `$${stats.avg_amount}` : '—' },
          { label: 'Avg Fraud Amount', value: stats.avg_fraud_amount != null ? `$${stats.avg_fraud_amount}` : '—', color: '#fda4af' },
          { label: 'Avg Genuine Amount', value: stats.avg_genuine_amount != null ? `$${stats.avg_genuine_amount}` : '—', color: '#6ee7b7' },
        ].map(s => (
          <Card key={s.label} className="flex flex-col justify-between h-20">
            <span className="text-xs text-zinc-400 font-medium">{s.label}</span>
            <span className="text-xl font-semibold tracking-tight" style={{ color: s.color || '#f4f4f5' }}>{s.value}</span>
          </Card>
        ))}
      </div>

      {/* Amount Distribution & Filters */}
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <SectionTitle subtitle="Frequency of transaction amounts by class">
            Amount Distribution
          </SectionTitle>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 text-xs text-zinc-400">
              <span>Range ($):</span>
              <input
                type="number"
                value={minAmount}
                onChange={e => setMinAmount(+e.target.value)}
                className="form-input text-xs py-1 px-2"
                style={{ width: 80 }}
              />
              <span>to</span>
              <input
                type="number"
                value={maxAmount}
                onChange={e => setMaxAmount(+e.target.value)}
                className="form-input text-xs py-1 px-2"
                style={{ width: 80 }}
              />
            </div>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={filteredDist} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="bin_start" tick={{ fontSize: 10, fill: '#71717a' }} tickFormatter={v => `$${v}`} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: '#71717a' }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
            <Bar dataKey="genuine_count" fill="#10b981" name="Genuine" radius={[2, 2, 0, 0]} opacity={0.8} />
            <Bar dataKey="fraud_count" fill="#f43f5e" name="Fraud" radius={[2, 2, 0, 0]} opacity={0.8} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Feature Importances */}
      {data.feature_importances?.length > 0 && (
        <Card>
          <SectionTitle subtitle="Contribution weight calculated from Random Forest splits">
            Key Feature Weights
          </SectionTitle>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={data.feature_importances} layout="vertical" margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis type="number" tick={{ fontSize: 10, fill: '#71717a' }} axisLine={false} tickLine={false} />
              <YAxis dataKey="feature" type="category" tick={{ fontSize: 10, fill: '#a1a1aa' }} width={38} axisLine={false} tickLine={false} />
              <Tooltip formatter={(v) => v.toFixed(5)} />
              <Bar dataKey="importance" name="Importance" radius={[0, 2, 2, 0]}>
                {data.feature_importances.map((_, i) => (
                  <Cell key={i} fill="#60a5fa" opacity={1 - (i / data.feature_importances.length) * 0.55} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Anomalies List */}
      <Card>
        <div className="flex items-center justify-between mb-3">
          <SectionTitle subtitle="Outliers identified via unsupervised Isolation Forest">
            Unsupervised Anomalies
          </SectionTitle>
          <div className="flex gap-1 bg-[#17181c] p-0.5 rounded border border-white/[0.06]">
            {['all', 'fraud', 'genuine'].map(f => (
              <button
                key={f}
                onClick={() => setClassFilter(f)}
                className={`text-[10px] px-2.5 py-0.5 rounded capitalize transition-colors cursor-pointer ${
                  classFilter === f ? 'bg-white/[0.1] text-zinc-100 font-medium' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {anomalies.length === 0 ? (
          <EmptyState title="No anomalies found" description="Adjust criteria or review loaded dataset." />
        ) : (
          <div className="overflow-x-auto max-h-52">
            <table className="data-table">
              <thead>
                <tr><th>Row</th><th>Amount</th><th>Anomaly Score</th><th>Ground Truth</th></tr>
              </thead>
              <tbody>
                {anomalies.slice(0, 100).map(a => (
                  <tr key={a.row_index}>
                    <td className="font-mono text-xs">{a.row_index}</td>
                    <td className="font-medium text-zinc-200">${a.amount.toFixed(2)}</td>
                    <td className="font-mono text-xs" style={{ color: a.anomaly_score < -0.2 ? '#fda4af' : '#fcd34d' }}>
                      {a.anomaly_score.toFixed(4)}
                    </td>
                    <td>
                      {a.is_fraud === null ? '—' : <Badge type={a.is_fraud ? 'fraud' : 'genuine'}>{a.is_fraud ? 'Fraud' : 'Genuine'}</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
