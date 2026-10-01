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
 * Creates the {@link Invocation} for an auxiliary call the SDK makes on a
 * request's behalf (HITL classifier, steering, goal judge, web-fetch analyst).
 * The call's tokens count toward the request's usage, but the request's limits
 * do not apply to it and its turns do not count against them. Returns
 * `undefined` when there is no enclosing request, so the call runs standalone.
 *
 * @param parent - The enclosing request's invocation, if any
 * @returns An invocation sharing the request's usage, or `undefined`
 * @internal
 */
export function createAuxiliaryInvocation(parent: Invocation | undefined): Invocation | undefined {
  if (!parent) return undefined
  return { usage: parent.usage, turns: 0, limits: undefined }
}

/**
 * Links each object the SDK passes to extension code (a hook event, tool
 * context, model-call or strategy context, reduce options) to its request's
 * {@link Invocation}. Module-private so the state never appears on a public
 * type or at runtime on the object itself. The link is by object identity: a
 * copy or spread of the object does not have it.
 */
const invocationsByObject = new WeakMap<object, Invocation>()

/**
 * Links the request's {@link Invocation} to an object the SDK passes to
 * extension code, so SDK code that later receives that object can join the
 * request with {@link readInvocation}.
 *
 * @param target - The object to link the request to
 * @param invocation - The request-scoped state to link; `undefined` links nothing
 * @internal
 */
export function linkInvocation(target: object, invocation: Invocation | undefined): void {
  if (invocation !== undefined) invocationsByObject.set(target, invocation)
}

/**
 * Returns the {@link Invocation} linked to an object by {@link linkInvocation}.
 *
 * @param source - An object the SDK passed to extension code, if any
 * @returns The linked request-scoped state, or `undefined` when there is none
 * @internal
 */
export function readInvocation(source: object | undefined): Invocation | undefined {
  return source === undefined ? undefined : invocationsByObject.get(source)
}
