import type { InvokeLimits } from '../types/agent.js'
import type { StopReason } from '../types/messages.js'
import { accumulateUsage, createEmptyUsage, type Usage } from '../models/streaming.js'

declare const agentContextBrand: unique symbol

/**
 * Handle to the request an object belongs to.
 *
 * A top-level `invoke()` / `stream()` call starts a request, and the SDK attaches
 * its handle to the objects it passes to hooks, tools, and extension points. Pass
 * it to {@link InvokeOptions.agentContext} to run a nested call in the same
 * request, sharing its limits and usage.
 *
 * Only the SDK can create one, and its type exposes no state.
 */
export interface AgentContext {
  readonly [agentContextBrand]: true
}

/**
 * The state behind an {@link AgentContext}: a request's limits, turns, and usage.
 *
 * Every nested call that joins the request shares one instance. A nested call
 * that sets its own limits, and an auxiliary call, get a child instance instead.
 * A child adds its usage to its parent; a nested call's child is also bound by
 * the parent's limits, while no limits apply to an auxiliary call.
 *
 * @internal
 */
export class AgentContextImplementation implements AgentContext {
  declare readonly [agentContextBrand]: true

  /** Turns taken under this instance, including its nested calls. */
  turns = 0

  /** Token usage of every model call made under this instance, including its children. */
  readonly usage: Usage = createEmptyUsage()

  /**
   * @param limits - Limits checked against this instance's turns and usage; `undefined` for none
   * @param _parent - The parent instance, which also receives this one's usage, if any
   * @param _auxiliary - Whether this is an auxiliary call: the parent's limits don't apply and its turns don't count toward the parent
   */
  constructor(
    readonly limits: InvokeLimits | undefined,
    private readonly _parent?: AgentContextImplementation,
    private readonly _auxiliary = false
  ) {}

  /** The parent whose limits also apply to this instance, if any. */
  get limitingParent(): AgentContextImplementation | undefined {
    return this._auxiliary ? undefined : this._parent
  }

  /** Usage of the whole request, read from the top-level instance. */
  get requestUsage(): Usage {
    return this._parent?.requestUsage ?? this.usage
  }

  /**
   * Adds one model call's usage to this instance and every parent.
   *
   * @param usage - The usage the model call reported
   */
  recordUsage(usage: Usage): void {
    accumulateUsage(this.usage, usage)
    this._parent?.recordUsage(usage)
  }

  /** Counts one turn here and in every parent whose limits apply to this instance. */
  recordTurn(): void {
    this.turns += 1
    this.limitingParent?.recordTurn()
  }
}

/**
 * Creates the state for a new request or, given a parent, for a nested call that
 * sets its own limits. A child's usage and turns also count toward the parent,
 * and the parent's limits also apply to it.
 *
 * @param limits - Limits to apply, or `undefined` for none
 * @param parent - The parent instance, if any
 * @returns A new instance with no usage or turns yet
 * @internal
 */
export function createAgentContext(
  limits?: InvokeLimits,
  parent?: AgentContextImplementation
): AgentContextImplementation {
  return new AgentContextImplementation(limits, parent)
}

/**
 * Creates the {@link AgentContext} for an auxiliary call: the HITL classifier,
 * steering, goal judge, and web-fetch analyst, and the handle passed to extension
 * points (context strategies, conversation managers, routing strategies, memory
 * extractors). Its tokens add to the request's usage, but no limits apply to it
 * and its turns don't count toward the request. Returns `undefined` outside a
 * request, so the call runs standalone.
 *
 * @param parent - The handle of the request the call is made for, if any
 * @returns A handle whose usage adds to the request's, or `undefined`
 * @internal
 */
export function createAuxiliaryAgentContext(parent: AgentContext | undefined): AgentContext | undefined {
  const parentContext = unwrapAgentContext(parent)
  return parentContext && new AgentContextImplementation(undefined, parentContext, true)
}

/**
 * Returns the stop reason for the first limit reached, checking this instance's
 * own limits first and then each parent's, or `undefined` if none is reached.
 * Within one instance the order is turns, then totalTokens, then outputTokens.
 *
 * @param agentContext - The state to check
 * @returns The limit stop reason, or `undefined` while no limit is reached
 * @internal
 */
export function reachedLimit(agentContext: AgentContextImplementation): StopReason | undefined {
  const parent = agentContext.limitingParent
  return reachedOwnLimit(agentContext) ?? (parent && reachedLimit(parent))
}

function reachedOwnLimit(agentContext: AgentContextImplementation): StopReason | undefined {
  const limits = agentContext.limits
  if (!limits) return undefined

  const { outputTokens, totalTokens } = agentContext.usage
  if (limits.turns !== undefined && agentContext.turns >= limits.turns) {
    return 'limitTurns'
  }
  if (limits.totalTokens !== undefined && totalTokens >= limits.totalTokens) {
    return 'limitTotalTokens'
  }
  if (limits.outputTokens !== undefined && outputTokens >= limits.outputTokens) {
    return 'limitOutputTokens'
  }
  return undefined
}

/**
 * Returns the state behind an {@link AgentContext} handle.
 *
 * @param agentContext - A handle the SDK passed out, if any
 * @returns The state behind it, or `undefined` when there is no handle
 * @throws TypeError if the handle was not created by the SDK
 * @internal
 */
export function unwrapAgentContext(agentContext: AgentContext | undefined): AgentContextImplementation | undefined {
  if (agentContext === undefined) return undefined
  if (!(agentContext instanceof AgentContextImplementation))
    throw new TypeError('agentContext was not created by the SDK')
  return agentContext
}
