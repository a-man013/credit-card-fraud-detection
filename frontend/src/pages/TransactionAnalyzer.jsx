import { useState, useRef } from 'react'
import { Upload, Search, Download, Play, FileText, CheckCircle, AlertTriangle } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import { predictSingle, predictBatch, uploadDataset } from '../api/client'
import { Card, SectionTitle, Badge, Spinner, AlertBanner } from '../components/UI'

const SAMPLE_FEATURES = {
  Time: 0, V1: -1.359807, V2: -0.072781, V3: 2.536347, V4: 1.378155, V5: -0.338321,
  V6: 0.462388, V7: 0.239599, V8: 0.098698, V9: 0.363787, V10: 0.090794,
  V11: -0.551600, V12: -0.617801, V13: -0.991390, V14: -0.311169, V15: 1.468177,
  V16: -0.470401, V17: 0.207971, V18: 0.025791, V19: 0.403993, V20: 0.251412,
  V21: -0.018307, V22: 0.277838, V23: -0.110474, V24: 0.066928, V25: 0.128539,
  V26: -0.189115, V27: 0.133558, V28: -0.021053, Amount: 149.62
}

function SingleForm() {
  const [model, setModel] = useState('Random Forest')
  const [features, setFeatures] = useState(
    Object.fromEntries(Object.entries(SAMPLE_FEATURES).map(([k, v]) => [k, String(v)]))
  )
  const [result, setResult] = useState(null)

  const mutation = useMutation({
    mutationFn: () => {
      const parsed = Object.fromEntries(Object.entries(features).map(([k, v]) => [k, parseFloat(v) || 0]))
      return predictSingle(parsed, model).then(r => r.data)
    },
    onSuccess: setResult,
  })

  const isFraud = result?.predicted_class === 1

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <label className="text-xs text-zinc-400">Model:</label>
        <select value={model} onChange={e => setModel(e.target.value)}
          className="form-input text-xs" style={{ width: 180 }}>
          <option>Random Forest</option>
          <option>Logistic Regression</option>
          <option>Decision Tree</option>
        </select>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 max-h-56 overflow-y-auto pr-1">
        {Object.entries(features).map(([k, v]) => (
          <div key={k}>
            <label className="text-[11px] font-mono text-zinc-400 block mb-0.5">{k}</label>
            <input className="form-input text-xs py-1 px-2" value={v}
              onChange={e => setFeatures(f => ({ ...f, [k]: e.target.value }))} />
          </div>
        ))}
      </div>

      <div className="flex gap-2 pt-1">
        <button onClick={() => mutation.mutate()} disabled={mutation.isPending}
          className="btn-primary flex items-center gap-1.5 text-xs">
          {mutation.isPending ? <><Spinner size={13} /> Evaluating…</> : <><Play size={13} /> Predict</>}
        </button>
        <button onClick={() => setFeatures(Object.fromEntries(Object.entries(SAMPLE_FEATURES).map(([k, v]) => [k, String(v)])))}
          className="btn-secondary text-xs">Reset Values</button>
      </div>

      {mutation.isError && <AlertBanner type="error">{mutation.error.message}</AlertBanner>}

      {result && (
        <div className="rounded-lg p-3.5 border transition-all" style={{
          background: isFraud ? 'rgba(244,63,94,0.06)' : 'rgba(16,185,129,0.06)',
          borderColor: isFraud ? 'rgba(244,63,94,0.2)' : 'rgba(16,185,129,0.2)',
        }}>
          <div className="flex items-center gap-2 mb-2">
            {isFraud ? <AlertTriangle size={16} className="text-rose-400" /> : <CheckCircle size={16} className="text-emerald-400" />}
            <span className="text-sm font-semibold tracking-tight" style={{ color: isFraud ? '#fda4af' : '#6ee7b7' }}>
              {result.prediction} Transaction
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
            <div><span className="text-zinc-400">Model: </span><span className="text-zinc-200">{result.model_used}</span></div>
            {result.probability_fraud != null && (
              <div><span className="text-zinc-400">Fraud Prob: </span>
                <span className={isFraud ? 'text-rose-300 font-medium' : 'text-emerald-300 font-medium'}>
                  {(result.probability_fraud * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {result.confidence && (
              <div><span className="text-zinc-400">Confidence: </span>
                <span className="text-zinc-200">{result.confidence}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function BatchUpload() {
  const [file, setFile] = useState(null)
  const [model, setModel] = useState('Random Forest')
  const [preview, setPreview] = useState(null)
  const [results, setResults] = useState(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const fileRef = useRef()

  const uploadMut = useMutation({
    mutationFn: () => uploadDataset(file).then(r => r.data),
    onSuccess: setPreview,
  })

  const predictMut = useMutation({
    mutationFn: () => predictBatch(file, model).then(r => r.data),
    onSuccess: setResults,
  })

  const downloadCSV = () => {
    if (!results) return
    const header = 'row_index,amount,prediction,predicted_class,probability_fraud,actual_class,is_correct\n'
    const rows = results.predictions.map(r =>
      `${r.row_index},${r.features?.Amount ?? ''},${r.prediction},${r.predicted_class},${r.probability_fraud ?? ''},${r.actual_class ?? ''},${r.is_correct ?? ''}`
    ).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'predictions.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  const filteredResults = results?.predictions?.filter(r => {
    const matchFilter = filter === 'all' || (filter === 'fraud' && r.predicted_class === 1) || (filter === 'genuine' && r.predicted_class === 0)
    const matchSearch = search === '' || String(r.row_index).includes(search) || String(r.features?.Amount ?? '').includes(search)
    return matchFilter && matchSearch
  }) || []

  return (
    <div className="space-y-4">
      {/* Upload area */}
      <div
        className="border border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors bg-[#121316]/40 hover:border-white/20"
        style={{ borderColor: file ? 'rgba(255,255,255,0.3)' : 'var(--border-color)' }}
        onClick={() => fileRef.current.click()}
      >
        <input ref={fileRef} type="file" accept=".csv" className="hidden"
          onChange={e => { setFile(e.target.files[0]); setPreview(null); setResults(null) }} />
        <Upload size={20} className="text-zinc-500 mx-auto mb-2" />
        <p className="text-xs font-medium text-zinc-300">
          {file ? file.name : 'Click or drop a CSV file to batch score'}
        </p>
        <p className="text-[11px] text-zinc-400 mt-0.5">
          Standard Kaggle format (Time, V1–V28, Amount, Class)
        </p>
      </div>

      {file && (
        <div className="flex flex-wrap gap-2 items-center">
          <select value={model} onChange={e => setModel(e.target.value)} className="form-input text-xs" style={{ width: 170 }}>
            <option>Random Forest</option>
            <option>Logistic Regression</option>
            <option>Decision Tree</option>
          </select>
          <button onClick={() => uploadMut.mutate()} disabled={uploadMut.isPending} className="btn-secondary flex items-center gap-1.5 text-xs">
            {uploadMut.isPending ? <Spinner size={12} /> : <FileText size={12} />} Preview
          </button>
          <button onClick={() => predictMut.mutate()} disabled={predictMut.isPending} className="btn-primary flex items-center gap-1.5 text-xs">
            {predictMut.isPending ? <><Spinner size={12} /> Processing…</> : <><Play size={12} /> Run Batch</>}
          </button>
          {results && (
            <button onClick={downloadCSV} className="btn-secondary flex items-center gap-1.5 text-xs">
              <Download size={12} /> Export CSV
            </button>
          )}
        </div>
      )}

      {uploadMut.isError && <AlertBanner type="error">{uploadMut.error.message}</AlertBanner>}
      {predictMut.isError && <AlertBanner type="error">{predictMut.error.message}</AlertBanner>}

      {/* Preview */}
      {preview && (
        <Card>
          <SectionTitle subtitle={`${preview.rows.toLocaleString()} rows × ${preview.columns} features`}>
            Data Preview
          </SectionTitle>
          <div className="overflow-x-auto max-h-40">
            <table className="data-table">
              <thead>
                <tr>{Object.keys(preview.preview[0] || {}).slice(0, 8).map(k => <th key={k}>{k}</th>)}</tr>
              </thead>
              <tbody>
                {preview.preview.slice(0, 5).map((row, i) => (
                  <tr key={i}>{Object.values(row).slice(0, 8).map((v, j) => <td key={j}>{String(v).slice(0, 10)}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Prediction results */}
      {results && (
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
            <SectionTitle subtitle={`${results.fraud_count} fraud of ${results.total} (${results.fraud_rate}%)`}>
              Batch Score Output
            </SectionTitle>
            <div className="flex gap-2 items-center">
              <div className="relative">
                <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
                <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Filter index/amount…"
                  className="form-input pl-7 text-xs py-1" style={{ width: 140 }} />
              </div>
              <div className="flex gap-1 bg-[#17181c] p-0.5 rounded border border-white/[0.06]">
                {['all', 'genuine', 'fraud'].map(f => (
                  <button key={f} onClick={() => setFilter(f)}
                    className={`text-[10px] px-2 py-0.5 rounded capitalize transition-colors ${
                      filter === f ? 'bg-white/[0.1] text-zinc-100 font-medium' : 'text-zinc-400 hover:text-zinc-200'
                    }`}>
                    {f}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="overflow-x-auto max-h-64">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Row</th><th>Amount</th><th>Predicted</th><th>Fraud Prob.</th>
                  {results.has_actual_labels && <><th>Actual</th><th>Correct</th></>}
                </tr>
              </thead>
              <tbody>
                {filteredResults.slice(0, 150).map(r => (
                  <tr key={r.row_index}>
                    <td className="font-mono text-xs">{r.row_index}</td>
                    <td className="text-zinc-200 font-medium">${(r.features?.Amount ?? 0).toFixed(2)}</td>
                    <td><Badge type={r.predicted_class === 1 ? 'fraud' : 'genuine'}>{r.prediction}</Badge></td>
                    <td className="text-xs font-mono">{r.probability_fraud != null ? `${(r.probability_fraud * 100).toFixed(1)}%` : '—'}</td>
                    {results.has_actual_labels && <>
                      <td><Badge type={r.actual_class === 1 ? 'fraud' : 'genuine'}>{r.actual_class === 1 ? 'Fraud' : 'Genuine'}</Badge></td>
                      <td className={r.is_correct ? 'text-emerald-400 font-medium' : 'text-rose-400 font-medium'}>
                        {r.is_correct ? '✓' : '✗'}
                      </td>
                    </>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

export default function TransactionAnalyzer() {
  const [tab, setTab] = useState('single')

  return (
    <div className="page-enter space-y-4">
      <div className="flex gap-1 p-0.5 rounded-md bg-[#121316] border border-white/[0.06] inline-flex">
        {[['single', 'Single Prediction'], ['batch', 'Batch CSV']].map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`px-3 py-1.5 text-xs rounded font-medium transition-colors cursor-pointer ${
              tab === id ? 'bg-white/[0.1] text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <Card>
        <SectionTitle subtitle={tab === 'single' ? 'Input features for real-time inference' : 'Upload file to score thousands of records'}>
          {tab === 'single' ? 'Real-time Prediction' : 'Batch Scoring'}
        </SectionTitle>
        {tab === 'single' ? <SingleForm /> : <BatchUpload />}
      </Card>
    </div>
  )
}
