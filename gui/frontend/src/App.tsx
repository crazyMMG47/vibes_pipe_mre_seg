import { useState, useCallback, useEffect } from 'react'
import { SubjectSidebar } from './components/SubjectSidebar'
import { CandidateGrid } from './components/CandidateGrid'
import { SliceScrubber } from './components/SliceScrubber'
import { MetricsPanel } from './components/MetricsPanel'
import { StiffnessPanel } from './components/StiffnessPanel'
import { CandidateSelector } from './components/CandidateSelector'
import { StatusBar, type Toast } from './components/StatusBar'
import { ErrorBoundary } from './components/ErrorBoundary'
import { useSubject } from './hooks/useSubject'

let _toastId = 0

function MainPanel({ subjectId }: { subjectId: string }) {
  const { data: detail } = useSubject(subjectId)
  const [axis, setAxis] = useState(2)
  const [index, setIndex] = useState(-1)
  const [overlay, setOverlay] = useState<'gt' | 'none'>('none')
  const [selectedSample, setSelectedSample] = useState<number | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])

  // Reset state when subject changes
  useEffect(() => {
    setAxis(2)
    setIndex(-1)
    setOverlay('none')
    setSelectedSample(null)
  }, [subjectId])

  // Update document title
  useEffect(() => {
    document.title = `${subjectId} · vibes_pipe Viewer`
    return () => { document.title = 'vibes_pipe · Prediction Viewer' }
  }, [subjectId])

  const addToast = useCallback((type: Toast['type'], message: string) => {
    const id = ++_toastId
    setToasts(prev => [...prev, { id, type, message }])
  }, [])

  const dismissToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const nSamples = detail?.n_mc_samples ?? 0
  const maxIndex = detail?.prob_shape
    ? (detail.prob_shape.length >= 4 ? detail.prob_shape[axis + 1] - 1 : detail.prob_shape[axis] - 1)
    : 100

  // Keyboard scrubbing
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        setIndex(prev => Math.min(prev < 0 ? Math.floor(maxIndex / 2) + 1 : prev + 1, maxIndex))
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        setIndex(prev => Math.max(prev < 0 ? Math.floor(maxIndex / 2) - 1 : prev - 1, 0))
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [maxIndex])

  return (
    <div className="flex-1 flex flex-col min-h-0 p-4 gap-4 overflow-y-auto">
      {/* Subject header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-gray-100">{subjectId}</h1>
          <div className="text-xs text-gray-500">
            {detail?.scanner_type ?? '…'} · {nSamples} MC samples
            {detail?.saved_at ? ` · saved ${detail.saved_at.slice(0, 10)}` : ''}
          </div>
        </div>
        <CandidateSelector
          subjectId={subjectId}
          selectedSample={selectedSample}
          onExportSuccess={path => addToast('success', `Pseudo-GT written to ${path}`)}
          onExportError={msg => addToast('error', msg)}
        />
      </div>

      {/* Scrubber */}
      <SliceScrubber
        axis={axis}
        onAxisChange={setAxis}
        index={index}
        maxIndex={maxIndex}
        onIndexChange={setIndex}
        overlay={overlay}
        onOverlayChange={setOverlay}
      />

      {/* Grid + stiffness row */}
      <ErrorBoundary label="Candidate Grid">
        <div className="flex gap-4 items-start overflow-x-auto">
          {nSamples > 0 ? (
            <CandidateGrid
              subjectId={subjectId}
              nSamples={nSamples}
              axis={axis}
              index={index}
              overlay={overlay}
              selectedSample={selectedSample}
              onSelectSample={setSelectedSample}
            />
          ) : (
            <div className="text-gray-500 text-sm py-8">
              No MC samples found for this subject.
            </div>
          )}
          <ErrorBoundary label="Stiffness Panel">
            <StiffnessPanel
              subjectId={subjectId}
              available={detail?.stiffness_available ?? false}
              axis={axis}
              index={index}
            />
          </ErrorBoundary>
        </div>
      </ErrorBoundary>

      {/* Metrics */}
      <ErrorBoundary label="Metrics Panel">
        <MetricsPanel
          subjectId={subjectId}
          selectedSample={selectedSample}
          onSelectSample={setSelectedSample}
        />
      </ErrorBoundary>

      <StatusBar toasts={toasts} onDismiss={dismissToast} />
    </div>
  )
}

export default function App() {
  const [activeId, setActiveId] = useState<string | null>(null)

  return (
    <div className="flex h-screen overflow-hidden">
      <SubjectSidebar activeId={activeId} onSelect={setActiveId} />
      <main className="flex-1 overflow-hidden flex flex-col">
        {activeId ? (
          <MainPanel key={activeId} subjectId={activeId} />
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-600 gap-3">
            <div className="text-4xl">◈</div>
            <div className="text-lg font-semibold">vibes_pipe · Prediction Viewer</div>
            <div className="text-sm">Select a subject from the sidebar to begin.</div>
          </div>
        )}
      </main>
    </div>
  )
}
