import { useState } from 'react'
import { useSubjects } from '../hooks/useSubjects'
import type { SubjectSummary } from '../types'

function diceColor(dice: number | null): string {
  if (dice === null) return 'text-gray-500'
  if (dice >= 0.8) return 'text-emerald-400'
  if (dice >= 0.6) return 'text-amber-400'
  return 'text-red-400'
}

function ScannerBadge({ type }: { type: string }) {
  const t = type?.toUpperCase()
  const color = t === 'GE' ? 'bg-blue-900 text-blue-300' : t === 'SIEMENS' ? 'bg-violet-900 text-violet-300' : 'bg-gray-700 text-gray-400'
  return <span className={`text-xs px-1.5 py-0.5 rounded font-bold ${color}`}>{t || '?'}</span>
}

function SubjectRow({ s, active, onClick }: { s: SubjectSummary; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 rounded-lg flex flex-col gap-1 transition-colors
        ${active ? 'bg-indigo-900 border border-indigo-600' : 'hover:bg-gray-800 border border-transparent'}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold truncate">{s.id}</span>
        <ScannerBadge type={s.scanner_type} />
      </div>
      <div className="flex items-center gap-3 text-xs text-gray-400">
        <span className={`font-mono ${diceColor(s.dice)}`}>
          Dice {s.dice !== null ? s.dice.toFixed(3) : '—'}
        </span>
        <span>{s.n_mc_samples} samples</span>
        {s.stiffness_available
          ? <span className="text-teal-400" title="Stiffness available">◈ stiff</span>
          : <span className="text-gray-600" title="No stiffness yet">◈</span>}
      </div>
    </button>
  )
}

export function SubjectSidebar({
  activeId,
  onSelect,
}: {
  activeId: string | null
  onSelect: (id: string) => void
}) {
  const { data: subjects, isLoading, isError } = useSubjects()
  const [search, setSearch] = useState('')

  const filtered = (subjects ?? []).filter(s =>
    s.id.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <aside className="w-64 shrink-0 flex flex-col bg-gray-900 border-r border-gray-800 h-screen">
      <div className="p-3 border-b border-gray-800">
        <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2">Subjects</div>
        <input
          type="text"
          placeholder="Search…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full bg-gray-800 text-sm text-gray-200 px-2 py-1.5 rounded border border-gray-700 focus:outline-none focus:border-indigo-500"
        />
      </div>
      <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-1">
        {isLoading && (
          <div className="space-y-2 p-2">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-14 bg-gray-800 rounded-lg animate-pulse" />
            ))}
          </div>
        )}
        {isError && (
          <div className="p-3 text-red-400 text-sm">Failed to load subjects. Is the backend running?</div>
        )}
        {!isLoading && !isError && filtered.length === 0 && (
          <div className="p-3 text-gray-500 text-sm">
            {subjects?.length === 0
              ? 'No inference results found.\nRun inference first.'
              : 'No subjects match.'}
          </div>
        )}
        {filtered.map(s => (
          <SubjectRow key={s.id} s={s} active={s.id === activeId} onClick={() => onSelect(s.id)} />
        ))}
      </div>
    </aside>
  )
}
