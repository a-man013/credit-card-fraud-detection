// Shared UI primitives (Minimalist)

export function Card({ children, className = '', style = {} }) {
  return (
    <div className={`glass-card p-4 sm:p-5 ${className}`} style={style}>
      {children}
    </div>
  )
}

export function SectionTitle({ children, subtitle, action }) {
  return (
    <div className="flex items-start justify-between mb-4">
      <div>
        <h2 className="text-sm font-semibold tracking-tight text-zinc-100" style={{ margin: 0 }}>
          {children}
        </h2>
        {subtitle && <p className="text-xs text-zinc-400 mt-0.5">{subtitle}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}

export function Badge({ type = 'info', children }) {
  const cls = {
    fraud: 'badge-fraud',
    genuine: 'badge-genuine',
    medium: 'badge-medium',
    info: 'badge-info',
  }
  return <span className={`badge ${cls[type] || 'badge-info'}`}>{children}</span>
}

export function Spinner({ size = 16 }) {
  return (
    <div style={{
      width: size, height: size,
      border: '2px solid rgba(255,255,255,0.15)',
      borderTopColor: '#f4f4f5',
      borderRadius: '50%',
      animation: 'spin 0.7s linear infinite',
      flexShrink: 0,
    }} />
  )
}

export function LoadingCard({ rows = 3 }) {
  return (
    <div className="glass-card p-4 sm:p-5 space-y-2.5">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="shimmer h-3.5" style={{ width: `${60 + (i % 3) * 15}%` }} />
      ))}
    </div>
  )
}

export function EmptyState({ icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-2 text-center">
      {icon && <div className="text-zinc-600 mb-1">{icon}</div>}
      <p className="text-sm font-medium text-zinc-300">{title}</p>
      {description && <p className="text-xs text-zinc-400 max-w-xs">{description}</p>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  )
}

export function AlertBanner({ type = 'info', children }) {
  const styles = {
    info: { bg: 'rgba(255, 255, 255, 0.03)', border: 'rgba(255, 255, 255, 0.08)', color: '#d4d4d8' },
    warning: { bg: 'rgba(245, 158, 11, 0.07)', border: 'rgba(245, 158, 11, 0.18)', color: '#fde68a' },
    success: { bg: 'rgba(16, 185, 129, 0.07)', border: 'rgba(16, 185, 129, 0.18)', color: '#a7f3d0' },
    error: { bg: 'rgba(244, 63, 94, 0.07)', border: 'rgba(244, 63, 94, 0.18)', color: '#fecdd3' },
  }
  const s = styles[type] || styles.info
  return (
    <div className="rounded-md px-3.5 py-2.5 text-xs flex items-center gap-2"
      style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}>
      {children}
    </div>
  )
}

export function StatRow({ label, value, highlight }) {
  return (
    <div className="flex items-center justify-between py-2 border-b" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
      <span className="text-xs text-zinc-400">{label}</span>
      <span className="text-xs font-medium" style={{ color: highlight || 'var(--text-primary)' }}>{value}</span>
    </div>
  )
}

// Add spin keyframe dynamically
if (typeof document !== 'undefined') {
  const style = document.createElement('style')
  style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }'
  document.head.appendChild(style)
}
