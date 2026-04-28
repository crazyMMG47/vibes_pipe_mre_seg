import { Component, type ReactNode, type ErrorInfo } from 'react'

interface Props {
  children: ReactNode
  label?: string
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[ErrorBoundary: ${this.props.label ?? 'panel'}]`, error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="bg-red-950 border border-red-800 rounded-xl p-4 text-red-300 text-sm flex flex-col gap-1">
          <div className="font-bold">{this.props.label ?? 'Panel'} error</div>
          <div className="text-xs font-mono text-red-400 opacity-80 break-all">
            {this.state.error.message}
          </div>
          <button
            className="mt-2 text-xs text-red-400 hover:text-red-200 underline self-start"
            onClick={() => this.setState({ error: null })}
          >
            Retry
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
