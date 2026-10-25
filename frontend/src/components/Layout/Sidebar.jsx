import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Search, BarChart3, Lightbulb,
  Database, ChevronLeft, ChevronRight, Shield
} from 'lucide-react'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/analyzer', icon: Search, label: 'Transaction Analyzer' },
  { to: '/performance', icon: BarChart3, label: 'Model Performance' },
  { to: '/insights', icon: Lightbulb, label: 'Fraud Insights' },
  { to: '/dataset', icon: Database, label: 'Dataset Management' },
]

export default function Sidebar({ collapsed, setCollapsed }) {
  return (
    <aside
      className="sidebar-transition flex flex-col flex-shrink-0 h-screen sticky top-0 z-30"
      style={{
        width: collapsed ? 56 : 216,
        background: '#0c0d10',
        borderRight: '1px solid var(--border-color)',
      }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-3.5 border-b"
        style={{ borderColor: 'var(--border-color)', height: 52 }}>
        <div className="flex-shrink-0 w-7 h-7 rounded-md flex items-center justify-center bg-white/[0.08] border border-white/[0.1]">
          <Shield size={14} className="text-zinc-100" />
        </div>
        {!collapsed && (
          <div className="flex items-center gap-1.5 overflow-hidden">
            <span className="font-semibold text-xs tracking-tight text-zinc-100">FraudShield</span>
            <span className="text-[10px] uppercase font-mono px-1 py-0.2 rounded bg-white/[0.06] text-zinc-400"></span>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto overflow-x-hidden">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${isActive
                ? 'bg-white/[0.08] text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.03]'
              }`
            }
            title={collapsed ? label : undefined}
          >
            <Icon size={15} className="flex-shrink-0" />
            {!collapsed && <span className="truncate">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse button */}
      <div className="p-2 border-t" style={{ borderColor: 'var(--border-color)' }}>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center justify-center w-full py-1.5 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04] transition-colors cursor-pointer text-xs"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={14} /> : (
            <div className="flex items-center gap-2 w-full px-2">
              <ChevronLeft size={14} />
              <span>Collapse</span>
            </div>
          )}
        </button>
      </div>
    </aside>
  )
}
