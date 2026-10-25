import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import {
  CreditCard, ShieldAlert, ShieldCheck, TrendingUp,
  RefreshCw, Database
} from 'lucide-react'
import { getDashboard } from '../api/client'
import { Card, SectionTitle, Badge, AlertBanner } from '../components/UI'

const COLORS_PIE = ['#10b981', '#f43f5e']

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <p className="text-xs font-medium text-zinc-300 mb-1">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 text-xs">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: p.color || p.fill }} />
          <span className="text-zinc-400">{p.name}:</span>
          <span className="font-medium text-zinc-100">{p.value?.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

function KPICard({ icon: Icon, label, value, sub, highlightColor }) {
  return (
    <Card className="flex flex-col justify-between h-28">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-400">{label}</span>
        <Icon size={15} className="text-zinc-500" />
      </div>
      <div>
        <p className="text-2xl font-semibold tracking-tight text-zinc-100">{value}</p>
        {sub && <p className="text-[11px] text-zinc-400 mt-0.5">{sub}</p>}
      </div>
    </Card>
  )
}

export default function Dashboard() {
  const [statusFilter, setStatusFilter] = useState('all')

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['dashboard'],
    queryFn: () => getDashboard().then(r => r.data),
    refetchInterval: 60000,
  })

  if (isLoading) return (
    <div className="page-enter space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="glass-card p-4 h-28 shimmer" />
        ))}
      </div>
    </div>
  )

  if (error) return (
    <div className="page-enter">
      <AlertBanner type="error">
        Failed to load dashboard: {error.message}. Make sure backend is running on port 8000.
      </AlertBanner>
    </div>
  )

  const { kpi, time_series, recent_transactions, fraud_alerts, system_status, is_sample_data } = data

  const pieData = [
    { name: 'Genuine', value: kpi.genuine_count },
    { name: 'Fraud', value: kpi.fraud_count },
  ]

  const filteredTxns = (recent_transactions || []).filter(t => {
    if (statusFilter === 'all') return true
    if (statusFilter === 'fraud') return t.prediction === 'Fraud'
    return t.prediction === 'Genuine'
  })

  return (
    <div className="page-enter space-y-5">
      {is_sample_data && (
        <AlertBanner type="warning">
          <span>Sample synthetic dataset active. For production validation, upload a real dataset in <b>Dataset Management</b>.</span>
        </AlertBanner>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <KPICard
          icon={Database}
          label="Total Transactions"
          value={kpi.total_transactions.toLocaleString()}
          sub="Analyzed records"
        />
        <KPICard
          icon={ShieldCheck}
          label="Genuine Transactions"
          value={kpi.genuine_count.toLocaleString()}
          sub={`${(100 - kpi.fraud_rate).toFixed(1)}% safe`}
        />
        <KPICard
          icon={ShieldAlert}
          label="Fraud Detected"
          value={kpi.fraud_count.toLocaleString()}
          sub="Flagged transactions"
        />
        <KPICard
          icon={TrendingUp}
          label="Fraud Rate"
          value={`${kpi.fraud_rate.toFixed(2)}%`}
          sub="Of total volume"
        />
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Activity Trend */}
        <Card className="lg:col-span-2">
          <SectionTitle subtitle="Volume trends across timeline">Transaction Activity</SectionTitle>
          <ResponsiveContainer width="100%" height={210}>
            <LineChart data={time_series} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
              <Line type="monotone" dataKey="genuine" stroke="#10b981" strokeWidth={1.5} dot={false} name="Genuine" />
              <Line type="monotone" dataKey="fraud" stroke="#f43f5e" strokeWidth={1.5} dot={false} name="Fraud" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Class Distribution */}
        <Card>
          <SectionTitle subtitle="Share of genuine vs fraudulent">Class Ratio</SectionTitle>
          <ResponsiveContainer width="100%" height={210}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="45%" innerRadius={50} outerRadius={68}
                dataKey="value" paddingAngle={2} stroke="none">
                {pieData.map((_, i) => <Cell key={i} fill={COLORS_PIE[i]} />)}
              </Pie>
              <Tooltip formatter={(v) => v.toLocaleString()} />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Secondary Bar Chart */}
      <Card>
        <SectionTitle subtitle="Volume comparison per time period">Time Interval Comparison</SectionTitle>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={time_series} barGap={2} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
            <Bar dataKey="genuine" fill="#10b981" name="Genuine" radius={[2, 2, 0, 0]} opacity={0.85} />
            <Bar dataKey="fraud" fill="#f43f5e" name="Fraud" radius={[2, 2, 0, 0]} opacity={0.85} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Tables & Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Recent Transactions */}
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <SectionTitle subtitle="Last recorded stream">Recent Transactions</SectionTitle>
            <div className="flex gap-1 bg-[#17181c] p-0.5 rounded-md border border-white/[0.06]">
              {['all', 'genuine', 'fraud'].map(f => (
                <button
                  key={f}
                  onClick={() => setStatusFilter(f)}
                  className={`text-[11px] px-2.5 py-1 rounded capitalize transition-colors cursor-pointer ${
                    statusFilter === f
                      ? 'bg-white/[0.1] text-zinc-100 font-medium'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th><th>Amount</th><th>Status</th><th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {filteredTxns.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="text-center py-6 text-xs text-zinc-400">
                      No matching transactions
                    </td>
                  </tr>
                ) : filteredTxns.map(t => (
                  <tr key={t.id}>
                    <td className="font-mono text-xs">{t.id}</td>
                    <td className="font-medium text-zinc-200">${t.amount.toFixed(2)}</td>
                    <td>
                      <Badge type={t.prediction === 'Fraud' ? 'fraud' : 'genuine'}>
                        {t.prediction}
                      </Badge>
                    </td>
                    <td className="text-[11px] text-zinc-400 font-mono">{t.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Sidebar: Alerts & State */}
        <div className="space-y-3">
          {/* Fraud Alerts */}
          <Card>
            <SectionTitle subtitle="Transactions exceeding threshold">Flagged Alerts</SectionTitle>
            <div className="space-y-2">
              {fraud_alerts.length === 0 ? (
                <p className="text-xs text-zinc-400 py-2">No active fraud flags</p>
              ) : fraud_alerts.map(a => (
                <div key={a.id} className="flex items-center justify-between p-2.5 rounded-md bg-[#16171b] border border-white/[0.05]">
                  <div>
                    <p className="text-xs font-semibold text-rose-300">${a.amount.toFixed(2)}</p>
                    <p className="text-[10px] text-zinc-400 font-mono">{a.timestamp}</p>
                  </div>
                  <Badge type={a.severity === 'High' ? 'fraud' : 'medium'}>{a.severity}</Badge>
                </div>
              ))}
            </div>
          </Card>

          {/* System State */}
          <Card>
            <SectionTitle>Status</SectionTitle>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between py-1 border-b border-white/[0.04]">
                <span className="text-zinc-400">Model Trained</span>
                <span className="flex items-center gap-1.5 font-medium text-zinc-300">
                  <span className={`w-1.5 h-1.5 rounded-full ${system_status.model_trained ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                  {system_status.model_trained ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-white/[0.04]">
                <span className="text-zinc-400">Dataset</span>
                <span className="flex items-center gap-1.5 font-medium text-zinc-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Loaded
                </span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-white/[0.04]">
                <span className="text-zinc-400">Active Model</span>
                <span className="text-zinc-300 font-medium">{system_status.active_model || '—'}</span>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-zinc-400">Records</span>
                <span className="text-zinc-300 font-mono">{system_status.records_processed?.toLocaleString()}</span>
              </div>
            </div>
          </Card>

          {/* Quick Refresh */}
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="w-full flex items-center justify-center gap-2 btn-secondary text-xs py-2"
          >
            <RefreshCw size={13} className={isFetching ? 'animate-spin' : ''} />
            <span>{isFetching ? 'Syncing…' : 'Sync Metrics'}</span>
          </button>
        </div>
      </div>
    </div>
  )
}
