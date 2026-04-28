export interface SubjectSummary {
  id: string
  scanner_type: string
  dice: number | null
  n_mc_samples: number
  mean_entropy: number | null
  ged: number | null
  mean_std: number | null
  saved_at: string
  stiffness_available: boolean
}

export interface SubjectDetail extends SubjectSummary {
  prob_shape: number[] | null
  threshold: number
  checkpoint_path: string | null
}

export interface SliceParams {
  subjectId: string
  volume: string         // "raw" | "gt" | "mean" | "sample_0" | "stiffness"
  axis: number           // 0 | 1 | 2
  index: number          // -1 = middle
  overlay: 'gt' | 'none'
  threshold: number
}

export interface SampleMetric {
  sample_index: number
  dice: number | null
  entropy: number
  std: number
}

export interface PerSubjectMetrics {
  mean_dice: number | null
  mean_entropy: number | null
  ged: number | null
  mean_std: number | null
  per_sample: SampleMetric[]
}
