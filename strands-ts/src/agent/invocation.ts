import type { InvokeLimits } from '../types/agent.js'
import { createEmptyUsage, type Usage } from '../models/streaming.js'

/**
 * Request-scoped state threaded through one invocation.
 *
 * The root {@link Agent.stream} call creates one `Invocation` and shares it by
 * reference across every agent the request reaches — sub-agents invoked as
 * tools and multi-agent nodes see the same instance.
 *
 * @internal
 */
export interface Invocation {
  /** Running token usage for the whole request, accumulated in place across every model call. */
  usage: Usage

  /** Count of agent-loop turns taken so far, across the whole request. */
  turns: number

  /** Limits inherited from the root request; `undefined` for none. */
  limits: InvokeLimits | undefined
}

/**
 * Creates a fresh {@link Invocation} with a zeroed usage total and no turns
 * taken yet.
 *
 * @param limits - Limits for the request, or `undefined` for none
 * @returns A new request-scoped state object
 * @internal
 */
export function createInvocation(limits?: InvokeLimits): Invocation {
  return { usage: createEmptyUsage(), turns: 0, limits }
}

/**
 * Derives an auxiliary view from the enclosing request's {@link Invocation}: it
 * shares the parent's `usage` by reference but drops the limits and keeps its
 * own turn count, so an auxiliary agent (steering, HITL, goal judge, web-fetch
 * analyst) adds its tokens to the request total without being limited by — or
 * counting turns against — it. Returns `undefined` when there is no enclosing
 * state, so the auxiliary call runs standalone.
 *
 * @param parent - The enclosing request's state, if any
 * @returns An auxiliary view sharing the parent's usage, or `undefined`
 * @internal
 */
export function deriveAuxiliaryInvocation(parent: Invocation | undefined): Invocation | undefined {
  if (!parent) return undefined
  return { usage: parent.usage, turns: 0, limits: undefined }
}

/**
 * Stamps the request's {@link Invocation} onto a carrier non-enumerably, so a
 * consumer that receives the carrier (a hook event, {@link ToolContext}, model
 * context) can read the request-scoped state — without the field surfacing in
 * enumeration or serialization of the carrier.
 *
 * @param carrier - The object to attach the state to
 * @param invocation - The request-scoped state to attach
 * @internal
 */
export function attachInvocation(carrier: object, invocation: Invocation): void {
  Object.defineProperty(carrier, 'invocation', {
    value: invocation,
    enumerable: false,
    configurable: true,
  })
}
