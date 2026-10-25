import { useState, useRef } from 'react'
import { Upload, Database, RotateCcw } from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { uploadDataset, getDatasetStatus, resetDataset } from '../api/client'
import { Card, SectionTitle, Badge, Spinner, AlertBanner } from '../components/UI'

function ProgressBar({ value, max, color }) {
  const pct = Math.min(100, max > 0 ? (value / max) * 100 : 0)
  return (
    <div className="h-1 rounded-full mt-1 bg-[#18191d] overflow-hidden">
      <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: color }} />
    </div>
  )
}

export default function DatasetManagement() {
  const [file, setFile] = useState(null)
  const [stats, setStats] = useState(null)
  const fileRef = useRef()
  const queryClient = useQueryClient()

  // Query currently loaded active dataset from backend
  const { data: currentDataset, refetch: refetchStatus } = useQuery({
    queryKey: ['datasetStatus'],
    queryFn: () => getDatasetStatus().then(r => r.data),
  })

  const uploadMut = useMutation({
    mutationFn: () => uploadDataset(file).then(r => r.data),
    onSuccess: (data) => {
      setStats(data)
      refetchStatus()
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      queryClient.invalidateQueries({ queryKey: ['health'] })
    },
  })

  const resetMut = useMutation({
    mutationFn: () => resetDataset().then(r => r.data),
    onSuccess: () => {
      setFile(null)
      setStats(null)
      refetchStatus()
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      queryClient.invalidateQueries({ queryKey: ['health'] })
    },
  })

  const handleDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f?.name.endsWith('.csv')) { setFile(f); setStats(null) }
  }

  return (
    <div className="page-enter space-y-4">
      {/* Current Active Dataset Status Card */}
      {currentDataset && (
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <Database size={15} className="text-zinc-400" />
                <span className="text-xs font-semibold text-zinc-100">Active Loaded Dataset:</span>
                <span className="text-xs font-mono text-zinc-300">{currentDataset.filename}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                  currentDataset.is_sample_data
                    ? 'bg-amber-500/10 text-amber-300 border border-amber-500/20'
                    : 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20'
                }`}>
                  {currentDataset.is_sample_data ? 'Synthetic Demo Data' : 'Persisted on Disk'}
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-1">
                {currentDataset.rows?.toLocaleString()} rows • {currentDataset.columns} features • {currentDataset.fraud_count?.toLocaleString()} fraud ({currentDataset.fraud_rate}%)
                {currentDataset.upload_time && ` • Uploaded at ${currentDataset.upload_time}`}
              </p>
            </div>

            {!currentDataset.is_sample_data && (
              <button
                onClick={() => resetMut.mutate()}
                disabled={resetMut.isPending}
                className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1.5 text-zinc-400 hover:text-zinc-200"
                title="Reset back to the default synthetic demo dataset"
              >
                <RotateCcw size={12} />
                <span>{resetMut.isPending ? 'Resetting…' : 'Reset to Demo Data'}</span>
              </button>
            )}
          </div>
        </Card>
      )}

      {/* Upload Zone */}
      <Card>
        <SectionTitle subtitle="Upload Kaggle creditcard.csv or compatible fraud dataset (persists across sessions)">
          Upload New Dataset
        </SectionTitle>
        <div
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current.click()}
          className="border border-dashed rounded-lg p-7 text-center cursor-pointer transition-colors bg-[#121316]/40 hover:border-white/20"
          style={{ borderColor: file ? 'rgba(255,255,255,0.3)' : 'var(--border-color)' }}
        >
          <input ref={fileRef} type="file" accept=".csv" className="hidden"
            onChange={e => { setFile(e.target.files[0]); setStats(null) }} />
          <Upload size={22} className="text-zinc-500 mx-auto mb-2" />
          <p className="text-xs font-medium text-zinc-300">
            {file ? file.name : 'Click or drop a CSV dataset here to replace active data'}
          </p>
          <p className="text-[11px] text-zinc-400 mt-0.5">
            Expected: Time, V1–V28, Amount, Class (Kaggle ULB benchmark format)
          </p>
        </div>

        {file && (
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => uploadMut.mutate()}
              disabled={uploadMut.isPending}
              className="btn-primary flex items-center gap-1.5 text-xs"
            >
              {uploadMut.isPending ? <><Spinner size={12} /> Parsing & Persisting…</> : <><Upload size={12} /> Load Dataset</>}
            </button>
            <button
              onClick={() => { setFile(null); setStats(null) }}
              className="btn-secondary text-xs"
            >
              Remove
            </button>
          </div>
        )}
        {uploadMut.isError && <AlertBanner type="error" className="mt-2">{uploadMut.error.message}</AlertBanner>}
      </Card>

      {/* Dataset Statistics from freshly uploaded file */}
      {stats && (
        <>
          <div className="space-y-1.5">
            {stats.validation_messages.map((m, i) => (
              <AlertBanner key={i} type={m.startsWith('ERROR') ? 'error' : 'success'}>{m}</AlertBanner>
            ))}
            {stats.warnings.map((w, i) => (
              <AlertBanner key={i} type="warning">{w}</AlertBanner>
            ))}
          </div>

          {/* Quick Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: 'Total Rows', value: stats.rows.toLocaleString() },
              { label: 'Features', value: stats.columns },
              { label: 'Missing Cells', value: `${stats.missing_cells} (${stats.missing_pct}%)`, warn: stats.missing_cells > 0 },
              { label: 'Duplicates', value: stats.duplicate_rows, warn: stats.duplicate_rows > 0 },
            ].map(s => (
              <Card key={s.label} className="flex flex-col justify-between h-20">
                <span className="text-xs text-zinc-400 font-medium">{s.label}</span>
                <span className={`text-xl font-semibold tracking-tight ${s.warn ? 'text-amber-300' : 'text-zinc-100'}`}>
                  {s.value}
                </span>
              </Card>
            ))}
          </div>

          {/* Class Ratio Breakdown */}
          {stats.class_distribution && (
            <Card>
              <SectionTitle subtitle="Distribution of genuine (0) vs fraudulent (1) target classes">
                Class Imbalance Breakdown
              </SectionTitle>
              <div className="space-y-3">
                {Object.entries(stats.class_distribution).map(([cls, cnt]) => (
                  <div key={cls}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-zinc-400">
                        {cls === '0' ? 'Genuine (Class 0)' : 'Fraudulent (Class 1)'}
                      </span>
                      <span className="font-mono text-zinc-200">
                        {cnt.toLocaleString()} ({((cnt / stats.rows) * 100).toFixed(2)}%)
                      </span>
                    </div>
                    <ProgressBar value={cnt} max={stats.rows} color={cls === '0' ? '#10b981' : '#f43f5e'} />
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Column Details */}
          <Card>
            <SectionTitle subtitle="Schema definition and null value summary">
              Schema Inspection
            </SectionTitle>
            <div className="overflow-x-auto max-h-56">
              <table className="data-table">
                <thead>
                  <tr><th>Feature</th><th>Data Type</th><th>Null Count</th><th>Null %</th></tr>
                </thead>
                <tbody>
                  {stats.column_info.map(col => (
                    <tr key={col.name}>
                      <td className="font-mono text-xs text-zinc-200">{col.name}</td>
                      <td><span className="badge badge-info">{col.dtype}</span></td>
                      <td className={`font-mono text-xs ${col.null_count > 0 ? 'text-amber-300' : 'text-zinc-400'}`}>
                        {col.null_count}
                      </td>
                      <td className={`font-mono text-xs ${col.null_pct > 5 ? 'text-rose-300' : 'text-zinc-400'}`}>
                        {col.null_pct}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Table Preview */}
          <Card>
            <SectionTitle subtitle="Initial records from loaded tabular data">
              Record Sample
            </SectionTitle>
            <div className="overflow-x-auto max-h-48">
              <table className="data-table">
                <thead>
                  <tr>{Object.keys(stats.preview[0] || {}).slice(0, 10).map(k => <th key={k}>{k}</th>)}</tr>
                </thead>
                <tbody>
                  {stats.preview.slice(0, 8).map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).slice(0, 10).map((v, j) => (
                        <td key={j} className="font-mono text-xs text-zinc-300">{String(v).slice(0, 12)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* Guide Card */}
      <Card>
        <SectionTitle subtitle="Steps to acquire real transaction data">
          Kaggle Benchmark Dataset
        </SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-zinc-400">
          <div className="p-3 rounded-md bg-[#16171b] border border-white/[0.05]">
            <span className="font-mono text-zinc-200 block mb-1">01 / Download</span>
            Obtain <b>creditcard.csv</b> from Kaggle (ULB Machine Learning Group, 284,807 transactions).
          </div>
          <div className="p-3 rounded-md bg-[#16171b] border border-white/[0.05]">
            <span className="font-mono text-zinc-200 block mb-1">02 / Ingestion</span>
            Upload the CSV file into the drop zone above. It is validated and saved to disk automatically.
          </div>
          <div className="p-3 rounded-md bg-[#16171b] border border-white/[0.05]">
            <span className="font-mono text-zinc-200 block mb-1">03 / Model Training</span>
            Navigate to <b>Model Performance</b> and click <b>Train Models</b> to generate metrics.
          </div>
        </div>
      </Card>
    </div>
  )
}
