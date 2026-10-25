import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import AppLayout from './components/Layout/AppLayout'
import Dashboard from './pages/Dashboard'
import TransactionAnalyzer from './pages/TransactionAnalyzer'
import ModelPerformance from './pages/ModelPerformance'
import FraudInsights from './pages/FraudInsights'
import DatasetManagement from './pages/DatasetManagement'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="analyzer" element={<TransactionAnalyzer />} />
            <Route path="performance" element={<ModelPerformance />} />
            <Route path="insights" element={<FraudInsights />} />
            <Route path="dataset" element={<DatasetManagement />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
