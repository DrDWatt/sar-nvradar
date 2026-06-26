import React, { useState } from 'react'

const API_BASE = '/api'

// All algorithms — NVRadar image formation + GPU enhancement
// requiresPhaseHistory marks algorithms that need raw radar data
const ALL_ALGORITHMS = [
  { value: 'pfa', label: 'NVRadar Polar Format (PFA)', requiresPhaseHistory: true },
  { value: 'bp', label: 'NVRadar Backprojection (BP)', requiresPhaseHistory: true },
  { value: 'enhanced', label: 'NVRadar Enhanced GMTI', requiresGmti: true },
  { value: 'adaptive', label: 'GPU Adaptive Enhancement', requiresPhaseHistory: false },
  { value: 'lee', label: 'GPU Lee Speckle Filter', requiresPhaseHistory: false },
  { value: 'histogram_eq', label: 'GPU Histogram Equalization', requiresPhaseHistory: false },
]

export default function ImageViewer({ datasets, health }) {
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [selectedFile, setSelectedFile] = useState(0)
  const [algorithm, setAlgorithm] = useState('pfa')
  const [imageSize, setImageSize] = useState(512)
  const [loading, setLoading] = useState(false)
  const [originalUrl, setOriginalUrl] = useState(null)
  const [enhancedUrl, setEnhancedUrl] = useState(null)
  const [processingTime, setProcessingTime] = useState(null)
  const [agentAnalysis, setAgentAnalysis] = useState(null)

  const getAlgorithms = () => {
    const type = selectedDataset?.type
    return ALL_ALGORITHMS.filter(a => {
      if (a.requiresGmti) return type === 'gmti'
      if (a.requiresPhaseHistory) return type === 'target_discrimination'
      // GPU enhancement algorithms available for all types
      return true
    })
  }

  // Build data-specific query params
  const _dataParams = (ds) => {
    const baseDir = ds.base_dir ? `&base_dir=${encodeURIComponent(ds.base_dir)}` : ''
    if (ds.type === 'target_discrimination') {
      const vehicle = ds.id.split('/')[1]
      return `data_type=target_discrimination&vehicle=${vehicle}&file_index=${selectedFile}&image_size=${imageSize}${baseDir}`
    } else if (ds.type === 'gmti') {
      const channel = ds.id.split('/')[1]
      return `data_type=gmti&channel=${channel}&image_size=${imageSize}${baseDir}`
    } else {
      const path = ds.id.replace('geotiff/', '')
      return `data_type=geotiff&path=${encodeURIComponent(path)}&image_size=${imageSize}${baseDir}`
    }
  }

  const processImages = () => {
    if (!selectedDataset) return
    setLoading(true)
    setProcessingTime(null)
    setAgentAnalysis(null)

    const ds = selectedDataset
    const baseDir = ds.base_dir ? `&base_dir=${encodeURIComponent(ds.base_dir)}` : ''

    // --- Auto (AI Agent) mode ---
    if (algorithm === 'auto') {
      const params = _dataParams(ds)
      // Set original image URL
      let origUrl
      if (ds.type === 'target_discrimination') {
        const vehicle = ds.id.split('/')[1]
        origUrl = `${API_BASE}/image/target-disc/original?vehicle=${vehicle}&file_index=${selectedFile}&image_size=${imageSize}${baseDir}`
      } else if (ds.type === 'gmti') {
        const channel = ds.id.split('/')[1]
        origUrl = `${API_BASE}/image/gmti/original?channel=${channel}&image_size=${imageSize}${baseDir}`
      } else {
        const path = ds.id.replace('geotiff/', '')
        origUrl = `${API_BASE}/image/geotiff/original?path=${encodeURIComponent(path)}&image_size=${imageSize}${baseDir}`
      }
      setOriginalUrl(origUrl)
      setEnhancedUrl(null)

      // Run agent analysis (POST) then fetch best image (GET)
      fetch(`${API_BASE}/auto-enhance?${params}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          setAgentAnalysis(data)
          setProcessingTime(data.total_time_ms)
          // Now fetch the best image
          return fetch(`${API_BASE}/auto-enhance/image?${params}`)
        })
        .then(r => r.blob())
        .then(blob => {
          setEnhancedUrl(URL.createObjectURL(blob))
          setLoading(false)
        })
        .catch(() => setLoading(false))
      return
    }

    // --- Manual algorithm mode ---
    let origUrl, enhUrl

    if (ds.type === 'target_discrimination') {
      const vehicle = ds.id.split('/')[1]
      origUrl = `${API_BASE}/image/target-disc/original?vehicle=${vehicle}&file_index=${selectedFile}&image_size=${imageSize}${baseDir}`
      enhUrl = `${API_BASE}/image/target-disc/enhanced?vehicle=${vehicle}&file_index=${selectedFile}&algorithm=${algorithm}&image_size=${imageSize}${baseDir}`
    } else if (ds.type === 'gmti') {
      const channel = ds.id.split('/')[1]
      origUrl = `${API_BASE}/image/gmti/original?channel=${channel}&image_size=${imageSize}${baseDir}`
      enhUrl = `${API_BASE}/image/gmti/enhanced?channel=${channel}&image_size=${imageSize}${baseDir}`
    } else if (ds.type === 'geotiff') {
      const path = ds.id.replace('geotiff/', '')
      origUrl = `${API_BASE}/image/geotiff/original?path=${encodeURIComponent(path)}&image_size=${imageSize}${baseDir}`
      enhUrl = `${API_BASE}/image/geotiff/enhanced?path=${encodeURIComponent(path)}&algorithm=${algorithm}&image_size=${imageSize}${baseDir}`
    }

    setOriginalUrl(origUrl)
    setEnhancedUrl(null)

    const startTime = performance.now()
    fetch(enhUrl)
      .then(r => {
        const procTime = r.headers.get('X-Processing-Time-Ms')
        if (procTime) setProcessingTime(parseFloat(procTime))
        return r.blob()
      })
      .then(blob => {
        setEnhancedUrl(URL.createObjectURL(blob))
        if (!processingTime) {
          setProcessingTime(Math.round(performance.now() - startTime))
        }
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }

  return (
    <div>
      {/* Controls */}
      <div className="bg-gray-800 rounded-lg p-4 mb-6 border border-gray-700">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Dataset selector */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Dataset</label>
            <select
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
              value={selectedDataset ? selectedDataset.id : ''}
              onChange={e => {
                const ds = datasets.find(d => d.id === e.target.value)
                setSelectedDataset(ds || null)
                setSelectedFile(ds ? (ds.best_index || 0) : 0)
                // Set best default algorithm for selected type
                if (ds) {
                  if (ds.type === 'target_discrimination') setAlgorithm('pfa')
                  else if (ds.type === 'gmti') setAlgorithm('enhanced')
                  else setAlgorithm('adaptive')
                }
              }}
            >
              <option value="">Select dataset...</option>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} ({ds.file_count} file{ds.file_count > 1 ? 's' : ''})
                </option>
              ))}
            </select>
          </div>

          {/* Look angle / file selector */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              File
              {selectedDataset?.best_index !== undefined && selectedFile === selectedDataset.best_index && (
                <span className="ml-1 text-green-400 text-xs">(best SNR)</span>
              )}
            </label>
            {selectedDataset && selectedDataset.files?.length > 1 ? (
              <select
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                value={selectedFile}
                onChange={e => setSelectedFile(parseInt(e.target.value))}
              >
                {selectedDataset.files.map(f => (
                  <option key={f.index} value={f.index}>
                    {f.angle}{f.index === selectedDataset.best_index ? ' (best)' : ''}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={selectedDataset?.files?.[0]?.name || 'N/A'}
                disabled
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm opacity-50"
              />
            )}
          </div>

          {/* Algorithm */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Algorithm</label>
            <select
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
              value={algorithm}
              onChange={e => setAlgorithm(e.target.value)}
            >
              <option value="auto">Auto (AI Agent)</option>
              <optgroup label="NVRadar Algorithms">
                {getAlgorithms().filter(a => a.requiresPhaseHistory || a.requiresGmti).map(a => (
                  <option key={a.value} value={a.value}>{a.label}</option>
                ))}
              </optgroup>
              <optgroup label="GPU Enhancement">
                {getAlgorithms().filter(a => !a.requiresPhaseHistory && !a.requiresGmti).map(a => (
                  <option key={a.value} value={a.value}>{a.label}</option>
                ))}
              </optgroup>
            </select>
          </div>

          {/* Process button */}
          <div className="flex items-end">
            <button
              onClick={processImages}
              disabled={!selectedDataset || loading}
              className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded px-4 py-2 text-sm transition-colors"
            >
              {loading ? (algorithm === 'auto' ? 'Agent Analyzing...' : 'Processing...') : (algorithm === 'auto' ? 'Run AI Agent' : 'Process SAR Data')}
            </button>
          </div>
        </div>

        {processingTime && (
          <div className="mt-3 text-sm text-gray-400">
            Processing time: <span className="text-green-400 font-mono">{processingTime} ms</span>
            {health?.gpu_available && <span className="ml-2 text-green-500">(GPU accelerated)</span>}
          </div>
        )}
      </div>

      {/* Side-by-side comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Original */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-700">
            <h2 className="text-lg font-semibold text-orange-400">Original (Range-Doppler)</h2>
            <p className="text-xs text-gray-400 mt-1">
              {selectedDataset?.type === 'geotiff' ? 'Raw GeoTIFF log-scaled' : 'Raw phase history \u2192 2D FFT'}
            </p>
          </div>
          <div className="p-4 flex items-center justify-center min-h-[400px]">
            {originalUrl ? (
              <img src={originalUrl} alt="Original SAR" className="max-w-full max-h-[512px] border border-gray-600" />
            ) : (
              <div className="text-gray-500 text-center">
                <p className="text-4xl mb-2">📡</p>
                <p>Select a dataset and click Process</p>
              </div>
            )}
          </div>
        </div>

        {/* Enhanced */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-700">
            <h2 className="text-lg font-semibold text-green-400">
              {algorithm === 'auto'
                ? `Agent Selected: ${agentAnalysis?.best_algorithm_name || 'Analyzing...'}`
                : ['pfa', 'bp', 'enhanced'].includes(algorithm)
                  ? 'Enhanced (NVRadar Pipeline)'
                  : 'Enhanced (GPU Processing)'
              }
            </h2>
            <p className="text-xs text-gray-400 mt-1">
              {algorithm === 'auto'
                ? `NVIDIA Nemotron Agent \u00b7 ${agentAnalysis?.llm_model || 'loading...'}`
                : `${ALL_ALGORITHMS.find(a => a.value === algorithm)?.label || algorithm} \u00b7 GPU-accelerated`
              }
            </p>
          </div>
          <div className="p-4 flex items-center justify-center min-h-[400px]">
            {loading ? (
              <div className="text-center">
                <div className="animate-spin w-12 h-12 border-4 border-green-500 border-t-transparent rounded-full mx-auto mb-3"></div>
                <p className="text-gray-400">
                  {algorithm === 'auto' ? 'Agent running all algorithms & analyzing...' : `Running ${algorithm.toUpperCase()} on GPU...`}
                </p>
              </div>
            ) : enhancedUrl ? (
              <img src={enhancedUrl} alt="Enhanced SAR" className="max-w-full max-h-[512px] border border-gray-600" />
            ) : (
              <div className="text-gray-500 text-center">
                <p className="text-4xl mb-2">🎯</p>
                <p>Enhanced image will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Agent Analysis Panel */}
      {agentAnalysis && (
        <div className="mt-6 bg-gray-800 rounded-lg border border-green-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-green-700 bg-green-900/20">
            <h3 className="text-lg font-semibold text-green-400">NVRadar AI Agent Analysis</h3>
            <p className="text-xs text-gray-400 mt-1">
              Model: {agentAnalysis.llm_model} &middot; Total time: {agentAnalysis.total_time_ms} ms
            </p>
          </div>
          <div className="p-4">
            {/* LLM Recommendation */}
            <div className="mb-4">
              <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">{agentAnalysis.analysis}</p>
            </div>

            {/* Algorithm Comparison Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-gray-400 border-b border-gray-700">
                  <tr>
                    <th className="px-3 py-2">Algorithm</th>
                    <th className="px-3 py-2 text-right">SNR (dB)</th>
                    <th className="px-3 py-2 text-right">Contrast</th>
                    <th className="px-3 py-2 text-right">Sharpness</th>
                    <th className="px-3 py-2 text-right">Entropy</th>
                    <th className="px-3 py-2 text-right">Score</th>
                    <th className="px-3 py-2 text-right">Time (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {agentAnalysis.all_results?.map(r => (
                    <tr key={r.algorithm} className={`border-b border-gray-700/50 ${
                      r.is_best ? 'bg-green-900/20 text-green-300' : 'text-gray-300'
                    }`}>
                      <td className="px-3 py-2 font-medium">
                        {r.is_best && <span className="mr-1">&#9733;</span>}
                        {r.name}
                      </td>
                      <td className="px-3 py-2 text-right font-mono">{r.metrics.snr_db}</td>
                      <td className="px-3 py-2 text-right font-mono">{r.metrics.contrast}</td>
                      <td className="px-3 py-2 text-right font-mono">{r.metrics.sharpness}</td>
                      <td className="px-3 py-2 text-right font-mono">{r.metrics.entropy_bits}</td>
                      <td className="px-3 py-2 text-right font-mono font-bold">{r.metrics.composite_score}</td>
                      <td className="px-3 py-2 text-right font-mono">{r.processing_time_ms}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
