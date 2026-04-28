import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { useMetrics } from '../hooks/useMetrics'

function StatCard({ label, value, fmt }: { label: string; value: number | null | undefined; fmt?: (v: number) => string }) {
  const display = value != null ? (fmt ? fmt(value) : value.toFixed(4)) : '—'
  const color =
    label === 'Mean Dice'
      ? value == null ? 'text-gray-500' : value >= 0.8 ? 'text-emerald-400' : value >= 0.6 ? 'text-amber-400' : 'text-red-400'
      : 'text-cyan-400'
  return (
    <div className="flex flex-col items-center bg-gray-900 border border-gray-700 rounded-xl px-5 py-3 min-w-32">
      <span className="text-xs text-gray-500 uppercase tracking-widest mb-1">{label}</span>
      <span className={`text-lg font-mono font-bold ${color}`}>{display}</span>
    </div>
  )
}

function barColor(dice: number | null): string {
  if (dice == null) return '#4b5563'
  if (dice >= 0.8) return '#34d399'
  if (dice >= 0.6) return '#fbbf24'
  return '#f87171'
}

interface Props {
  subjectId: string | null
  selectedSample: number | null
  onSelectSample: (k: number) => void
}

export function MetricsPanel({ subjectId, selectedSample, onSelectSample }: Props) {
  const { data, isLoading, isError } = useMetrics(subjectId)

  if (!subjectId) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-col gap-4">
      <div className="text-xs font-bold text-gray-400 uppercase tracking-widest">Metrics</div>

      {isLoading && <div className="h-32 bg-gray-800 rounded-lg animate-pulse" />}

      {isError && (
        <div className="text-gray-500 text-sm">Metrics unavailable (no GT at inference time or metrics not computed).</div>
      )}

      {data && (
        <>
          <div className="flex gap-3 flex-wrap">
            <StatCard label="Mean Dice" value={data.mean_dice} />
            <StatCard label="Mean Entropy" value={data.mean_entropy} fmt={v => v.toExponential(2)} />
            <StatCard label="GED" value={data.ged} fmt={v => v.toExponential(2)} />
          </div>

          {data.per_sample.length > 0 && (
            <div>
              <div className="text-xs text-gray-500 mb-2">Per-sample Dice — click bar to select candidate</div>
              <ResponsiveContainer width="100%" height={140}>
                <BarChart
                  data={data.per_sample.map(s => ({ name: `S${s.sample_index}`, diceVal: s.dice ?? 0, ...s }))}
                  onClick={e => {
                    if (e?.activePayload?.[0]) {
                      const idx = (e.activePayload[0].payload as { sample_index: number }).sample_index
                      onSelectSample(idx)
                    }
                  }}
                >
                  <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 11 }} width={32} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                    content={({ payload }) => {
                      if (!payload?.length) return null
                      const p = payload[0].payload as { dice: number | null; entropy: number; std: number; sample_index: number }
                      return (
                        <div className="text-xs bg-gray-900 border border-gray-700 rounded-lg p-2">
                          <div className="font-bold text-gray-300 mb-1">Sample {p.sample_index}</div>
                          <div>Dice: <span className="text-emerald-400">{p.dice?.toFixed(4) ?? '—'}</span></div>
                          <div>Entropy: {p.entropy.toExponential(2)}</div>
                          <div>Std: {p.std.toFixed(4)}</div>
                        </div>
                      )
                    }}
                  />
                  <Bar dataKey="diceVal" radius={[4, 4, 0, 0]} cursor="pointer">
                    {data.per_sample.map((s) => (
                      <Cell
                        key={s.sample_index}
                        fill={barColor(s.dice)}
                        opacity={selectedSample === s.sample_index ? 1 : 0.7}
                        stroke={selectedSample === s.sample_index ? '#f59e0b' : 'transparent'}
                        strokeWidth={selectedSample === s.sample_index ? 2 : 0}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  )
}
