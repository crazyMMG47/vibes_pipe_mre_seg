import { useEffect } from 'react'

export type Toast = { id: number; type: 'success' | 'error'; message: string }

interface Props {
  toasts: Toast[]
  onDismiss: (id: number) => void
}

export function StatusBar({ toasts, onDismiss }: Props) {
  return (
    <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50 pointer-events-none">
      {toasts.map(t => (
        <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  )
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(toast.id), 4000)
    return () => clearTimeout(timer)
  }, [toast.id, onDismiss])

  return (
    <div
      className={`pointer-events-auto max-w-sm px-4 py-2.5 rounded-xl shadow-xl text-sm font-medium
        flex items-start gap-2 border
        ${toast.type === 'success'
          ? 'bg-emerald-950 border-emerald-700 text-emerald-200'
          : 'bg-red-950 border-red-700 text-red-200'}`}
    >
      <span className="mt-0.5 shrink-0">{toast.type === 'success' ? '✓' : '✕'}</span>
      <span className="break-all">{toast.message}</span>
    </div>
  )
}
