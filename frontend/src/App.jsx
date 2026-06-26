import React, { useState, useEffect, useCallback } from 'react'
import DataPanel from './components/DataPanel'
import ImageViewer from './components/ImageViewer'

const API_BASE = '/api'

function App() {
  const [datasets, setDatasets] = useState([])
  const [sources, setSources] = useState([])
  const [health, setHealth] = useState(null)
  const [datasetsLoading, setDatasetsLoading] = useState(true)

  const fetchDatasets = useCallback(() => {
    setDatasetsLoading(true)
    fetch(`${API_BASE}/datasets`)
      .then(r => r.json())
      .then(d => {
        setDatasets(d.datasets || [])
        setDatasetsLoading(false)
      })
      .catch(() => {
        setDatasets([])
        setDatasetsLoading(false)
      })
  }, [])

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'error', gpu_available: false }))

    fetch(`${API_BASE}/sources`)
      .then(r => r.json())
      .then(d => setSources(d.sources || []))
      .catch(() => setSources([]))

    fetchDatasets()
  }, [fetchDatasets])

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div>
            <h1 className="text-2xl font-bold text-green-400">SAR Processing Viewer</h1>
            <p className="text-sm text-gray-400 mt-1">
              GPU-Accelerated Image Formation &middot; On-Demand Data Loading
            </p>
          </div>
          {health && (
            <div className="flex items-center gap-3">
              <span className={`inline-block w-3 h-3 rounded-full ${health.gpu_available ? 'bg-green-500' : 'bg-yellow-500'}`}></span>
              <span className="text-sm text-gray-300">
                {health.gpu_available ? 'GPU Active (CuPy)' : 'CPU Mode (NumPy)'}
              </span>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6">
        {/* Data management panel: upload, remote sources, dataset listing */}
        <DataPanel
          datasets={datasets}
          sources={sources}
          onDatasetsChange={fetchDatasets}
          datasetsLoading={datasetsLoading}
        />

        {/* Processing controls + side-by-side image viewer */}
        <ImageViewer datasets={datasets} health={health} />

        {/* Info */}
        <div className="mt-6 bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-2">About</h3>
          <p className="text-xs text-gray-400 leading-relaxed">
            Upload SAR data (zip with .mat/.tif files) or download from remote sources. Processing uses
            GPU-accelerated algorithms from the <span className="text-green-400">NVIDIA NVRadar</span> pipeline.
            Supports phase history (PFA/Backprojection), GMTI, and GeoTIFF SAR imagery (Lee filter,
            histogram equalization). CuPy GPU acceleration on NVIDIA DGX Spark (GB10, aarch64).
          </p>
        </div>
      </main>
    </div>
  )
}

export default App
