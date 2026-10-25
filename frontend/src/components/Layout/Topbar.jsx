import { useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getHealth } from '../../api/client'

const PAGE_TITLES = {
  '/': 'Dashboard',
  '/analyzer': 'Transaction Analyzer',
  '/performance': 'Model Performance',
  '/insights': 'Fraud Insights',
  '/dataset': 'Dataset Management',
}

export default function Topbar() {
  const { pathname } = useLocation()
  const title = PAGE_TITLES[pathname] || 'FraudShield'

  const { data: health, isError } = useQuery({
    queryKey: ['health'],
    queryFn: () => getHealth().then(r => r.data),
    refetchInterval: 30000,
    retry: 1,
  })

  const online = !isError && health?.status === 'ok'

  return (
    <header
      className="flex items-center justify-between px-5 border-b sticky top-0 z-20 backdrop-blur-md"
      style={{
        background: 'rgba(9, 9, 11, 0.85)',
        borderColor: 'var(--border-color)',
        height: 52,
      }}
    >
      <div className="flex items-center gap-2">
        <h1 className="text-xs font-medium tracking-tight text-zinc-200" style={{ margin: 0 }}>
          {title}
        </h1>
      </div>

      <div className="flex items-center gap-3">
        {/* API connection status */}
        <div className="flex items-center gap-1.5 text-[11px] text-zinc-400">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: online ? '#10b981' : '#f43f5e' }}
          />
          <span>{online ? 'API Connected' : 'API Disconnected'}</span>
        </div>

        {/* Model status */}
        {health && (
          <div className="flex items-center gap-1.5 text-[11px] text-zinc-400 pl-3 border-l"
            style={{ borderColor: 'var(--border-color)' }}>
            <span
              className={`w-1.5 h-1.5 rounded-full ${health.model_loaded ? 'pulse-dot' : ''}`}
              style={{ background: health.model_loaded ? '#60a5fa' : '#71717a' }}
            />
            <span>{health.model_loaded ? 'Model Ready' : 'Untrained'}</span>
          </div>
        )}
      </div>
    </header>
  )
}
