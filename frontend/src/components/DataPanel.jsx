import React, { useState } from 'react'

const API_BASE = '/api'

export default function DataPanel({ datasets, sources, onDatasetsChange, datasetsLoading }) {
  const [activeTab, setActiveTab] = useState('datasets')
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState('')
  const [downloading, setDownloading] = useState(null)
  const [downloadUrl, setDownloadUrl] = useState('')
  const [downloadName, setDownloadName] = useState('')

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setUploadMsg('')
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (res.ok) {
        setUploadMsg(`Uploaded "${file.name}" — ${data.datasets_found} dataset(s) found`)
        onDatasetsChange()
      } else {
        setUploadMsg(`Error: ${data.detail || 'Upload failed'}`)
      }
    } catch (err) {
      setUploadMsg(`Error: ${err.message}`)
    }
    setUploading(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) {
      const input = document.createElement('input')
      input.type = 'file'
      const dt = new DataTransfer()
      dt.items.add(file)
      input.files = dt.files
      handleFileUpload({ target: input })
    }
  }

  const handleDownloadFromUrl = async (url, sourceId, name) => {
    const targetUrl = url || downloadUrl
    if (!targetUrl) return

    setDownloading(sourceId || 'custom')
    try {
      const params = new URLSearchParams({ url: targetUrl })
      if (sourceId) params.append('source_id', sourceId)
      if (name || downloadName) params.append('dataset_name', name || downloadName)

      const res = await fetch(`${API_BASE}/download?${params}`, { method: 'POST' })
      const data = await res.json()
      if (res.ok) {
        setUploadMsg(`Downloaded — ${data.datasets_found} dataset(s) found`)
        onDatasetsChange()
        setDownloadUrl('')
        setDownloadName('')
      } else {
        setUploadMsg(`Error: ${data.detail || 'Download failed'}`)
      }
    } catch (err) {
      setUploadMsg(`Error: ${err.message}`)
    }
    setDownloading(null)
  }

  const tabs = [
    { id: 'datasets', label: `Datasets (${datasets.length})` },
    { id: 'upload', label: 'Upload' },
    { id: 'sources', label: 'Remote Sources' },
  ]

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 mb-6">
      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-green-400 border-b-2 border-green-400 bg-gray-750'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="p-4">
        {/* Status message */}
        {uploadMsg && (
          <div className={`mb-3 text-sm px-3 py-2 rounded ${
            uploadMsg.startsWith('Error') ? 'bg-red-900/30 text-red-300' : 'bg-green-900/30 text-green-300'
          }`}>
            {uploadMsg}
          </div>
        )}

        {/* Datasets tab */}
        {activeTab === 'datasets' && (
          <div>
            {datasetsLoading ? (
              <p className="text-gray-400 text-sm">Scanning for datasets...</p>
            ) : datasets.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-400 text-lg mb-2">No datasets loaded</p>
                <p className="text-gray-500 text-sm">Upload a zip file or download from a remote source to get started.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {datasets.map(ds => (
                  <div key={ds.id} className="bg-gray-700/50 rounded p-3 border border-gray-600">
                    <p className="text-sm font-medium text-gray-200 truncate">{ds.name}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      {ds.type === 'geotiff' ? 'GeoTIFF' : ds.type === 'gmti' ? 'GMTI' : 'Phase History'}
                      {' '}&middot; {ds.file_count} file(s)
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Upload tab */}
        {activeTab === 'upload' && (
          <div>
            <div
              onDrop={handleDrop}
              onDragOver={e => e.preventDefault()}
              className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-green-500 transition-colors"
            >
              {uploading ? (
                <div>
                  <div className="animate-spin w-8 h-8 border-3 border-green-500 border-t-transparent rounded-full mx-auto mb-3"></div>
                  <p className="text-gray-300">Uploading and extracting...</p>
                </div>
              ) : (
                <div>
                  <p className="text-3xl mb-2">📦</p>
                  <p className="text-gray-300 mb-2">Drag & drop a zip file here</p>
                  <p className="text-gray-500 text-sm mb-4">Supports .zip with .mat, .tif, .tiff, binary SAR data</p>
                  <label className="inline-block bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded text-sm cursor-pointer transition-colors">
                    Browse Files
                    <input type="file" accept=".zip,.tif,.tiff,.mat" className="hidden" onChange={handleFileUpload} />
                  </label>
                </div>
              )}
            </div>

            {/* Direct URL download */}
            <div className="mt-4 pt-4 border-t border-gray-700">
              <p className="text-sm font-medium text-gray-300 mb-2">Or download from URL:</p>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="https://example.com/sar-data.tif"
                  value={downloadUrl}
                  onChange={e => setDownloadUrl(e.target.value)}
                  className="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 placeholder-gray-500"
                />
                <input
                  type="text"
                  placeholder="Name (optional)"
                  value={downloadName}
                  onChange={e => setDownloadName(e.target.value)}
                  className="w-36 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 placeholder-gray-500"
                />
                <button
                  onClick={() => handleDownloadFromUrl()}
                  disabled={!downloadUrl || downloading}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded text-sm transition-colors"
                >
                  {downloading === 'custom' ? 'Downloading...' : 'Download'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Remote Sources tab */}
        {activeTab === 'sources' && (
          <div className="space-y-3">
            {sources.map(src => (
              <div key={src.id} className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-semibold text-gray-200">{src.name}</h4>
                      {src.auth_required && (
                        <span className="text-xs bg-yellow-800/50 text-yellow-300 px-2 py-0.5 rounded">Auth Required</span>
                      )}
                      <span className="text-xs bg-gray-600 text-gray-300 px-2 py-0.5 rounded">{src.format}</span>
                    </div>
                    <p className="text-xs text-gray-400 mt-1">{src.purpose}</p>
                    <p className="text-xs text-gray-500 mt-1">{src.description}</p>
                    {src.auth_required && src.auth_instructions && (
                      <p className="text-xs text-yellow-400/80 mt-1">{src.auth_instructions}</p>
                    )}
                  </div>
                  <a
                    href={src.docs_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-400 hover:text-blue-300 ml-4 whitespace-nowrap"
                  >
                    Docs &rarr;
                  </a>
                </div>

                {/* Sample downloads */}
                {src.samples.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-600">
                    <p className="text-xs text-gray-400 mb-2">Sample downloads:</p>
                    <div className="flex flex-wrap gap-2">
                      {src.samples.map((sample, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleDownloadFromUrl(sample.url, src.id, `${src.id}_${sample.name.replace(/\s+/g, '_')}`)}
                          disabled={downloading === src.id}
                          className="text-xs bg-blue-700/50 hover:bg-blue-600/50 text-blue-200 px-3 py-1 rounded transition-colors disabled:opacity-50"
                        >
                          {downloading === src.id ? '...' : sample.name}
                          {sample.size_mb && <span className="ml-1 text-blue-400">({sample.size_mb}MB)</span>}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
