import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { toInternal } from '../invocation.js'
import { BeforeModelCallEvent } from '../../hooks/events.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'

describe('AgentAsTool request-wide invocation', () => {
  it('folds the inner agent model usage into the parent request total', async () => {
    const innerModel = new MockMessageModel().addTurn(
      { type: 'textBlock', text: 'inner-done' },
      { usage: { inputTokens: 0, outputTokens: 60, totalTokens: 60 } }
    )
    const inner = new Agent({ model: innerModel, name: 'inner', description: 'inner agent' })

    let innerSawInvocation = false
    let innerSeenOutputTokens: number | undefined
    inner.addHook(BeforeModelCallEvent, (event) => {
      const invocation = toInternal(event.invocation)
      innerSawInvocation = invocation !== undefined
      innerSeenOutputTokens = invocation?.usage.outputTokens
    })

    const outerModel = new MockMessageModel()
      .addTurn([{ type: 'toolUseBlock', name: 'inner', toolUseId: 'tu-1', input: { input: 'hi' } }], {
        usage: { inputTokens: 0, outputTokens: 60, totalTokens: 60 },
      })
      .addTurn(
        { type: 'textBlock', text: 'outer-done' },
        { usage: { inputTokens: 0, outputTokens: 10, totalTokens: 10 } }
      )
    const outer = new Agent({ model: outerModel, tools: [inner.asTool()] })

    await outer.invoke('run inner')

    // Before the inner agent calls its model, the shared invocation already carries
    // the parent's first-turn output tokens — proof the same Invocation object was
    // threaded in, not a fresh per-agent one.
    expect(innerSawInvocation).toBe(true)
    expect(innerSeenOutputTokens).toBe(60)
  })

  it('trips a limit that only the parent-plus-child total exceeds', async () => {
    const innerModel = new MockMessageModel().addTurn(
      { type: 'textBlock', text: 'inner-done' },
      { usage: { inputTokens: 0, outputTokens: 60, totalTokens: 60 } }
    )
    const inner = new Agent({ model: innerModel, name: 'inner', description: 'inner agent' })

    const outerModel = new MockMessageModel()
      .addTurn([{ type: 'toolUseBlock', name: 'inner', toolUseId: 'tu-1', input: { input: 'hi' } }], {
        usage: { inputTokens: 0, outputTokens: 60, totalTokens: 60 },
      })
      .addTurn(
        { type: 'textBlock', text: 'outer-done' },
        { usage: { inputTokens: 0, outputTokens: 60, totalTokens: 60 } }
      )
    const outer = new Agent({ model: outerModel, tools: [inner.asTool()] })

    // Neither agent alone reaches 100 output tokens (each emits 60); only the shared
    // request total does, so tripping proves the limit spans parent and child.
    const result = await outer.invoke('run inner', { limits: { outputTokens: 100 } })

    expect(result.stopReason).toBe('limitOutputTokens')
  })
})
