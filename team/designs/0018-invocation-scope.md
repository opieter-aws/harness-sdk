# Invocation Scope: Tree-Wide Budget and Usage Accounting

**Status**: Proposed

**Date**: 2026-09-22

**Issue**: [#4299](https://github.com/strands-agents/harness-sdk/issues/4299)

**Builds on**: PR #4047 / issue
[#3863](https://github.com/strands-agents/harness-sdk/issues/3863), which gives every
SDK-owned auxiliary call (summarization, routing, extraction, web-fetch, HITL classifier,
goal judge, steering) a hook pair, per-source usage, and an OTel `source` dimension. This
design adds the thing that plumbing can't express: one shared budget and one authoritative
total across every model call a request causes, nested agents included.

**Scope**: TypeScript SDK first; the Python mirror follows.

## Problem

`Limits` and usage are scoped to a single `Agent`. But one request routinely makes other
model calls: a sub-agent invoked as a tool, a `Graph`/`Swarm` node, or an auxiliary call
like summarization or steering. None of them share a budget, and none of them roll up into
one total. When a sub-agent returns as tool output its result is flattened to a string, so
its usage is lost at the boundary.

A team building a managed harness can therefore neither cap the total work one request
causes nor report what it spent across the whole tree.

## Goals

- One turn/token budget over every model call a root invocation causes.
- Count usage as calls happen, so a later call sees what budget is left.
- Report one tree-wide total on the root result.
- Propagate through SDK-owned paths with no per-feature wiring.
- Leave each agent's own local metrics untouched.

Non-goals (v1): BidiAgent; a public third-party participation API; top-level structured
output; per-participant attribution (root vs. descendant vs. auxiliary).

## The `Invocation`

Introduce an `Invocation`: one object that represents a single request and holds three
things.

- A **running usage total** across every model call the request makes, tree-wide.
- An optional **budget** (`InvocationLimit`) that caps the whole tree.
- A set of **hooks** that fire for every agent in the tree.

```typescript
interface InvocationLimit {
  turns?: number
  outputTokens?: number
  totalTokens?: number
}

interface InvocationUsage {
  total: Usage // every model call the invocation caused, tree-wide
  turns: number // agent-loop turns across the tree
}

class Invocation {
  constructor(limit?: InvocationLimit)
  get usage(): InvocationUsage
  get limit(): InvocationLimit | undefined
  addHook<T extends HookableEvent>(eventType, callback, options?): HookCleanup
  // Internal: recordUsage, recordTurn, limitStopReason, applyLimit, invokeHooks.
}
```

One `Invocation` exists per request. The root `Agent.stream` creates one when the caller
doesn't supply it, and every SDK-owned nested call is handed the same object, so they share
its total and its budget. A caller who wants to set a tree-wide budget, read the total
across several calls, or register tree-wide hooks constructs one and passes it through
`InvokeOptions.invocation`; `InvokeOptions.limits` is the shorthand that builds one for the
budget-only case. Either way the tree total comes back on `AgentResult.usage`.

```typescript
invocation?: Invocation   // bring your own: set a budget, read the total across calls, add hooks
limits?: InvocationLimit  // shorthand that builds one for the budget-only case
```

We pass the object explicitly instead of keeping it in ambient storage. An ambient scope (a
`ContextVar`, or `AsyncLocalStorage` in TypeScript) is invisible at the call site, and
`AsyncLocalStorage` isn't available in the browser, which the SDK supports. SDK-owned paths
already thread their arguments through, so forwarding one more field is cheap: agent-as-tool
passes it on the options it already builds, and the auxiliary agents receive it the same
way.

**Tree-wide hooks.** A hook added on the invocation fires for every agent in the tree — the
union of each agent's own hooks and the invocation's. This is the only way to observe or
govern the whole request at once: a sub-agent invoked as a tool, a `Swarm`/`Graph` node, or
an SDK-owned auxiliary agent (HITL classifier, steering, web-fetch) has its own hook registry
the caller never holds a reference to, because the caller never constructs those agents.
Registering once on the shared invocation reaches all of them — for tree-wide tracing,
guardrails on every model call, or per-request cross-cutting state. It is the request-scoped
counterpart to `agent.hooks`, which sees only its own agent.

## Budget is opt-in; accounting is always on

Because the root always has an `Invocation`, `AgentResult.usage` always carries the tree
total. A budget only engages when the caller sets one. Caps are soft and checked at a turn
boundary: the loop stops at the top of the next turn once a cap is reached, never mid-call,
and tools requested by the previous turn always finish. On a tie the priority is turns, then
total tokens, then output tokens.

Nested and auxiliary agents share the same budget, but the SDK-owned auxiliary agents (HITL
classifier, steering, web-fetch, goal judge) fail safe: when the budget is already spent, an
agent that would otherwise be denied its one call still returns — a HITL check still asks
for approval, steering still proceeds — because `isInvocationLimitStopReason` lets the
caller tell a budget stop apart from a real answer. A spent budget shouldn't silence a
safety decision.

## Where usage recording and enforcement live

The `Invocation` owns the running total, but three jobs sit on the model-call path:
recording each call's usage into the invocation, enforcing the budget, and counting turns.
There are three ways to place them. They produce the same behavior; they differ in which
component does the work and in which surrounding API has to change.

| Responsibility                               | A: record in model base   | B: record in middleware       | C: model executor |
| -------------------------------------------- | ------------------------- | ----------------------------- | ----------------- |
| Record each call's usage                     | `Model.streamAggregated`  | `InvokeModelStage` middleware | `ModelExecutor`   |
| Accumulate the tree total                    | `Invocation`              | `Invocation`                  | `Invocation`      |
| Enforce the budget (stop)                    | agent loop                | agent loop                    | `ModelExecutor`   |
| Count turns                                  | agent loop                | agent loop                    | `ModelExecutor`   |
| Run the model call + retry                   | agent loop                | agent loop                    | `ModelExecutor`   |
| `Model` / `StreamOptions` carry `Invocation` | yes                       | no                            | no                |
| Direct (non-loop) calls counted              | yes, by passing the field | no, must route or record      | no, must route or record |

### A. Record in the model base

`Model.streamAggregated` records a call's usage into the invocation on a clean return. The
agent loop checks the budget at the top of each turn and counts the turn.

_API change:_ `StreamOptions` gains an `invocation?: Invocation` field, so the provider-facing
base type now references `Invocation`. Any caller of `streamAggregated` — including direct
calls outside the agent loop, like the routing classifier — is counted simply by passing the
field. That reach is the upside: recording sits at the one point every model call goes
through. The cost is the dependency a provider-facing type takes on.

### B. Record in model middleware

Recording moves off the `Model` base onto the `InvokeModelStage` middleware, which already
runs around every model call the loop makes and already carries the invocation on its
context. Enforcement and turn counting stay in the loop.

_API change:_ `StreamOptions` no longer carries `invocation`; the provider layer stops
knowing about it. In exchange, only calls that go through the loop's middleware are recorded.
Direct callers outside the loop are not counted unless they record explicitly or are routed
through the stage — today that is the routing classifier. (Summarization, memory extraction,
and compression also call the model directly, but they pass no invocation today, so they are
already uncounted and unaffected.)

### C. A model executor

Introduce an injectable `ModelExecutor` that mirrors `ToolExecutor` and owns the whole
model-call unit: recording, enforcement, and turn counting in one component. This is the most
centralized of the three. For it to count turns, one `execute` must be one turn, so the
model-retry loop moves into `execute` and brings the loop's continuation, redaction, routing,
and metric work with it. Enforcement returns a discriminated result (a stop reason or an
aggregated result) that the loop turns into an `AgentResult`, so a tripped budget stops
without throwing.

_API change:_ a new `ModelExecutor` abstraction plus a `modelExecutor?` field on
`AgentConfig` (with a runtime getter/setter), symmetric with `toolExecutor`. `Model` and
`StreamOptions` stay free of `Invocation`, and the agent loop no longer checks limits or
counts turns. Like B, direct calls outside the executor aren't counted unless routed through
it. The swap-in seam this provides — a caching, mocking, or recording proxy — already exists
as `InvokeModelStage` middleware, so the added abstraction earns its keep only if a
first-class model-call component is wanted for its own sake.

### Choosing

The options trade along two axes: whether a provider-facing type knows about `Invocation`,
and whether the model-call path becomes a distinct, swappable component or stays inline in
the loop. A counts every model call in the system for free, at the price of the provider-base
dependency. B keeps the provider layer clean but sees only calls that pass through the loop's
stage. C centralizes the three jobs into one component and mirrors the tool side, at the
price of relocating the retry loop and its orchestration.

## Example

```typescript
// One budget for the whole tree.
const result = await agent.invoke('Research the topic and summarize.', {
  limits: { turns: 20, totalTokens: 200_000 },
})
result.usage.total // every model call the request caused
result.usage.turns // turns across the tree

// Bring your own to carry the total across calls, or to register tree-wide hooks.
const invocation = new Invocation({ totalTokens: 200_000 })
await agent.invoke('Draft it.', { invocation })
await agent.invoke('Now revise.', { invocation })
invocation.usage.total
```

If the budget runs out inside a sub-agent, that agent stops and its partial answer returns
as a normal tool result; the root stops on its next turn check with a `limitTurns`,
`limitTotalTokens`, or `limitOutputTokens` stop reason. A plain `agent.invoke('hi')` returns
a total equal to the root's own usage and no cap.

## Consequences

One `limits` caps the whole request and one `usage` reports what it spent, with no
per-worker wiring.

A user's own nested `agent.invoke` inside a tool now joins the tree: its usage counts, and
with a budget set it can be stopped. A detached escape hatch is a possible follow-up.

Concurrent `Graph` fan-out can overshoot a soft cap slightly, as `Limits` already can.

Per-participant attribution isn't in v1 — the result reports one total and one turn count,
not a root/descendant/auxiliary breakdown. That can layer on later over #3863's `source`
dimension without changing this surface.

**Cross-SDK parity:** names carry to Python re-cased (`Invocation`, `InvocationUsage`,
`InvocationLimit`), and the shipped shape is the same in both.

**Follow-ups:** BidiAgent participation; a public third-party participation helper on the
auxiliary hook events #3863 exposes; top-level structured output; per-participant
attribution; a detached-subagent escape hatch.
