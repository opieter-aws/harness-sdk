import { unwrapAgentContext, type AgentContext } from '../agent/agent-context.js'
import type { ContentBlock, Message } from '../types/messages.js'
import type { Model, StreamAggregatedResult, StreamOptions } from './model.js'
import type { ModelStreamEvent } from './streaming.js'

/**
 * SDK-internal wrapper that adds a model call's token usage to its request. The
 * agent loop and every auxiliary call route their model calls through it. The
 * {@link AgentContext} is passed per call rather than on {@link StreamOptions},
 * so providers never see request state. Usage is added only when the call
 * returns normally and the provider reported it; a thrown error passes through
 * and adds nothing.
 *
 * @internal
 */
export class ModelProxy {
  private readonly _model: Model

  constructor(model: Model) {
    this._model = model
  }

  /**
   * Delegates to the wrapped model's `streamAggregated`, passing every event
   * through unchanged, then adds the call's usage to the request.
   *
   * @param messages - The conversation history to send to the model
   * @param options - Optional streaming configuration
   * @param agentContext - The request to add the usage to, or `undefined` to add it nowhere
   * @returns Async generator mirroring {@link Model.streamAggregated}
   */
  async *streamAggregated(
    messages: Message[],
    options?: StreamOptions,
    agentContext?: AgentContext
  ): AsyncGenerator<ModelStreamEvent | ContentBlock, StreamAggregatedResult, undefined> {
    const result = yield* this._model.streamAggregated(messages, options)
    this._record(result, agentContext)
    return result
  }

  /** Adds a completed call's reported usage to the request. */
  private _record(result: StreamAggregatedResult, agentContext?: AgentContext): void {
    const usage = result?.metadata?.usage
    const context = unwrapAgentContext(agentContext)
    if (usage !== undefined && context !== undefined) {
      context.recordUsage(usage)
    }
  }
}
