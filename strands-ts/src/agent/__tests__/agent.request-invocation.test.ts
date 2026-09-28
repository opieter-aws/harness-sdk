import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { createInvocation, deriveAuxiliaryInvocation } from '../invocation.js'
import { tool } from '../../tools/tool-factory.js'
import type { InternalInvokeOptions } from '../../types/agent.js'
import type { ToolContext } from '../../tools/tool.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'

describe('Agent request-wide limits', () => {
  it('leaves the parent limit intact and the child uncounted when a hand-written tool forwards invocationState', async () => {
    // A hand-written tool cannot join the request's shared state: the invocation
    // rides on explicit internal options, not on the invocationState bag. So a
    // sub-agent it runs on the forwarded bag executes standalone — the parent's
    // limit counts only the parent's own turns, and the child is never gated.
    const sub = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'sub done' },
        { usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }
      ),
      printer: false,
    })

    let subRuns = 0
    const relay = tool({
      name: 'relay',
      description: 'runs the sub-agent',
      callback: async (_input: unknown, context: ToolContext) => {
        const subResult = await sub.invoke('go', { invocationState: context.invocationState })
        // The child runs standalone, unaffected by the parent's limit.
        expect(subResult.stopReason).toBe('endTurn')
        subRuns += 1
        return 'ok'
      },
    })

    // Parent would loop indefinitely (every turn requests the tool) absent a limit.
    const parentModel = new MockMessageModel()
      .addTurn([{ type: 'toolUseBlock', name: 'relay', toolUseId: 'tu-1', input: {} }], {
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      })
      .addTurn([{ type: 'toolUseBlock', name: 'relay', toolUseId: 'tu-2', input: {} }], {
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      })
      .addTurn([{ type: 'toolUseBlock', name: 'relay', toolUseId: 'tu-3', input: {} }], {
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      })
    const parent = new Agent({ model: parentModel, tools: [relay], printer: false })

    const result = await parent.invoke('start', { limits: { turns: 2 } })

    // The limit trips on the parent's own second turn; the child ran to completion
    // on both turns because its work never counted against the limit.
    expect(result.stopReason).toBe('limitTurns')
    expect(parentModel.callCount).toBe(2)
    expect(subRuns).toBe(2)
  })

  it('produces a well-formed result when an inherited limit is already exhausted on entry', async () => {
    const agent = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'should not run' },
        { usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }
      ),
      printer: false,
    })

    // Inherit an enclosing request whose turn limit is already spent, so the loop
    // trips before its first message is appended.
    const inherited = createInvocation({ turns: 1 })
    inherited.turns = 5
    const result = await agent.invoke('go', { invocation: inherited } as InternalInvokeOptions)

    expect(result.stopReason).toBe('limitTurns')
    expect(() => result.toString()).not.toThrow()
    expect(result.toString()).toBe('')
  })

  it('does not throw when the invocationState bag is frozen', async () => {
    const agent = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'done' },
        { usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }
      ),
      printer: false,
    })

    const result = await agent.invoke('hi', { invocationState: Object.freeze({}) })

    expect(result.stopReason).toBe('endTurn')
  })

  it('throws when limits are set on a nested invoke', async () => {
    const agent = new Agent({
      model: new MockMessageModel().addTurn({ type: 'textBlock', text: 'done' }),
      printer: false,
    })

    // An inherited invocation marks this as a nested invoke; limits are root-only.
    await expect(
      agent.invoke('hi', { invocation: createInvocation(), limits: { turns: 1 } } as InternalInvokeOptions)
    ).rejects.toThrow(/limits can only be set on the root invoke/)
  })

  it('folds an auxiliary call into the request total without limiting it', async () => {
    // An enclosing request whose turn limit is already spent.
    const parentInvocation = createInvocation({ turns: 1 })
    parentInvocation.turns = 5
    parentInvocation.usage = { inputTokens: 1, outputTokens: 2, totalTokens: 3 }

    const aux = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'aux done' },
        { usage: { inputTokens: 4, outputTokens: 5, totalTokens: 9 } }
      ),
      printer: false,
    })

    const result = await aux.invoke('go', {
      invocation: deriveAuxiliaryInvocation(parentInvocation),
    } as InternalInvokeOptions)

    // The auxiliary agent runs to completion despite the spent limit...
    expect(result.stopReason).toBe('endTurn')
    // ...and its tokens fold into the shared request total.
    expect(parentInvocation.usage).toEqual({ inputTokens: 5, outputTokens: 7, totalTokens: 12 })
  })
})
