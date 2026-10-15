import { describe, expect, it, vi } from 'vitest'
import { Agent } from '../agent.js'
import {
  createAgentContext,
  createAuxiliaryAgentContext,
  AgentContextImplementation,
  unwrapAgentContext,
  type AgentContext,
} from '../agent-context.js'
import { AfterInvocationEvent, BeforeInvocationEvent, BeforeModelCallEvent } from '../../hooks/events.js'
import { tool } from '../../tools/tool-factory.js'
import type { ToolContext } from '../../tools/tool.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { Message, TextBlock } from '../../types/messages.js'
import { anyTrackingId } from '../../__fixtures__/message-helpers.js'

describe('Agent request-wide limits', () => {
  it('leaves the parent limit intact and the child uncounted when a hand-written tool forwards invocationState', async () => {
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
        expect(subResult.stopReason).toBe('endTurn')
        subRuns += 1
        return 'ok'
      },
    })

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
      messages: [
        new Message({ role: 'user', content: [new TextBlock('earlier question')] }),
        new Message({ role: 'assistant', content: [new TextBlock('earlier answer')] }),
      ],
      printer: false,
    })

    const inherited = createAgentContext({ turns: 1 })
    inherited.turns = 5
    const result = await agent.invoke('go', { agentContext: inherited })

    expect(result.stopReason).toBe('limitTurns')
    expect(result.lastMessage).toEqual(new Message({ role: 'assistant', content: [], trackingId: anyTrackingId }))
    expect(result.toString()).toBe('')
  })

  it('rejects invalid limits before any invocation hook runs', async () => {
    const agent = new Agent({ model: new MockMessageModel(), printer: false })
    const beforeInvocation = vi.fn()
    agent.addHook(BeforeInvocationEvent, beforeInvocation)

    await expect(agent.invoke('go', { limits: { turns: 0 } })).rejects.toThrow(TypeError)
    expect(beforeInvocation).not.toHaveBeenCalled()
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

  it("reports the whole request's usage on the result, while metrics cover this agent only", async () => {
    const inner = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'inner done' },
        { usage: { inputTokens: 5, outputTokens: 7, totalTokens: 12 } }
      ),
      name: 'inner',
      description: 'inner agent',
      printer: false,
    })
    const outer = new Agent({
      model: new MockMessageModel()
        .addTurn([{ type: 'toolUseBlock', name: 'inner', toolUseId: 'tu-1', input: { input: 'hi' } }], {
          usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
        })
        .addTurn(
          { type: 'textBlock', text: 'outer done' },
          { usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }
        ),
      tools: [inner.asTool()],
      printer: false,
    })

    const result = await outer.invoke('go')

    expect({ requestUsage: result.requestUsage, agentUsage: result.metrics?.accumulatedUsage }).toEqual({
      requestUsage: { inputTokens: 7, outputTokens: 10, totalTokens: 17 },
      agentUsage: expect.objectContaining({ inputTokens: 2, outputTokens: 3, totalTokens: 5 }),
    })
  })

  it("counts only a nested invoke's own tokens against its own token limits", async () => {
    const agent = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'done' },
        { usage: { inputTokens: 10, outputTokens: 10, totalTokens: 20 } }
      ),
      printer: false,
    })
    const enclosing = createAgentContext()
    enclosing.recordUsage({ inputTokens: 4000, outputTokens: 4000, totalTokens: 8000 })

    const result = await agent.invoke('go', { agentContext: enclosing, limits: { totalTokens: 5000 } })

    expect({ stopReason: result.stopReason, requestUsage: result.requestUsage }).toEqual({
      stopReason: 'endTurn',
      requestUsage: { inputTokens: 4010, outputTokens: 4010, totalTokens: 8020 },
    })
  })

  it("stops a nested invoke on the request's limit even when its own limit is looser", async () => {
    const loopTool = tool({ name: 'loop', description: 'loops', callback: () => 'again' })
    const model = new MockMessageModel()
      .addTurn([{ type: 'toolUseBlock', name: 'loop', toolUseId: 'tu-1', input: {} }])
      .addTurn([{ type: 'toolUseBlock', name: 'loop', toolUseId: 'tu-2', input: {} }])
    const agent = new Agent({ model, tools: [loopTool], printer: false })
    const request = createAgentContext({ turns: 2 })
    request.turns = 1

    const result = await agent.invoke('go', { agentContext: request, limits: { turns: 5 } })

    expect({ stopReason: result.stopReason, modelCalls: model.callCount, requestTurns: request.turns }).toEqual({
      stopReason: 'limitTurns',
      modelCalls: 1,
      requestTurns: 2,
    })
  })

  it('stops a nested invoke on its own tighter limit and counts its turns toward the request', async () => {
    const loopTool = tool({ name: 'loop', description: 'loops', callback: () => 'again' })
    const model = new MockMessageModel()
      .addTurn([{ type: 'toolUseBlock', name: 'loop', toolUseId: 'tu-1', input: {} }], {
        usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
      })
      .addTurn({ type: 'textBlock', text: 'never reached' })
    const agent = new Agent({ model, tools: [loopTool], printer: false })
    const enclosing = createAgentContext({ turns: 10 })

    const result = await agent.invoke('go', { agentContext: enclosing, limits: { turns: 1 } })

    expect(result.stopReason).toBe('limitTurns')
    expect(model.callCount).toBe(1)
    expect({ usage: enclosing.usage, turns: enclosing.turns }).toEqual({
      usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
      turns: 1,
    })
  })

  it('folds an auxiliary call into the request total without limiting it', async () => {
    const parentAgentContext = createAgentContext({ turns: 1 })
    parentAgentContext.recordUsage({ inputTokens: 1, outputTokens: 2, totalTokens: 3 })
    parentAgentContext.turns = 5

    const aux = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'aux done' },
        { usage: { inputTokens: 4, outputTokens: 5, totalTokens: 9 } }
      ),
      printer: false,
    })

    const result = await aux.invoke('go', {
      agentContext: createAuxiliaryAgentContext(parentAgentContext)!,
    })

    expect(result.stopReason).toBe('endTurn')
    expect(parentAgentContext.usage).toEqual({ inputTokens: 5, outputTokens: 7, totalTokens: 12 })
  })

  it('joins the request when a hand-written tool forwards context.agentContext', async () => {
    const sub = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'sub done' },
        { usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }
      ),
      printer: false,
    })

    const relay = tool({
      name: 'relay',
      description: 'runs the sub-agent',
      callback: async (_input: unknown, context: ToolContext) => {
        const subResult = await sub.invoke('go', {
          ...(context.agentContext && { agentContext: context.agentContext }),
        })
        return subResult.stopReason
      },
    })

    const parentModel = new MockMessageModel()
      .addTurn([{ type: 'toolUseBlock', name: 'relay', toolUseId: 'tu-1', input: {} }])
      .addTurn([{ type: 'toolUseBlock', name: 'relay', toolUseId: 'tu-2', input: {} }])
    const parent = new Agent({ model: parentModel, tools: [relay], printer: false })

    const result = await parent.invoke('start', { limits: { turns: 2 } })

    expect(result.stopReason).toBe('limitTurns')
    expect(parentModel.callCount).toBe(1)
  })

  it('attaches one request handle to the events and tool context it hands out', async () => {
    let probeContext: ToolContext | undefined
    const probe = tool({
      name: 'probe',
      description: 'captures its tool context',
      callback: (_input: unknown, context: ToolContext) => {
        probeContext = context
        return 'ok'
      },
    })
    const agent = new Agent({
      model: new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'probe', toolUseId: 'tu-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'done' }),
      tools: [probe],
      printer: false,
    })
    const modelCallEvents: BeforeModelCallEvent[] = []
    agent.addHook(BeforeModelCallEvent, (event) => {
      modelCallEvents.push(event)
    })

    await agent.invoke('go')

    const agentContext = modelCallEvents[0]?.agentContext
    expect(agentContext).toBeInstanceOf(AgentContextImplementation)
    expect(modelCallEvents[1]?.agentContext).toBe(agentContext)
    expect(probeContext?.agentContext).toBe(agentContext)
  })

  it('rejects an agent context the SDK did not create', async () => {
    const agent = new Agent({
      model: new MockMessageModel().addTurn({ type: 'textBlock', text: 'done' }),
      printer: false,
    })

    await expect(agent.invoke('hi', { agentContext: {} as AgentContext })).rejects.toThrow(
      'agentContext was not created by the SDK'
    )
  })

  it('starts fresh request state on each root invoke of a reused agent', async () => {
    const agent = new Agent({
      model: new MockMessageModel()
        .addTurn({ type: 'textBlock', text: 'first' })
        .addTurn({ type: 'textBlock', text: 'second' }),
      printer: false,
    })
    const modelCallEvents: BeforeModelCallEvent[] = []
    agent.addHook(BeforeModelCallEvent, (event) => {
      modelCallEvents.push(event)
    })

    await agent.invoke('one')
    await agent.invoke('two')

    const firstAgentContext = modelCallEvents[0]?.agentContext
    const secondAgentContext = modelCallEvents[1]?.agentContext
    expect(firstAgentContext).toBeDefined()
    expect(secondAgentContext).toBeDefined()
    expect(secondAgentContext).not.toBe(firstAgentContext)
    expect(unwrapAgentContext(secondAgentContext)?.turns).toBe(1)
  })

  it('ends on a limit stop even when a hook asks to resume', async () => {
    const loopTool = tool({ name: 'loop', description: 'loops', callback: () => 'again' })
    const model = new MockMessageModel().addTurn([{ type: 'toolUseBlock', name: 'loop', toolUseId: 'tu-1', input: {} }])
    const agent = new Agent({ model, tools: [loopTool], printer: false })
    let passes = 0
    let modelCalls = 0
    agent.addHook(BeforeInvocationEvent, () => {
      passes += 1
    })
    agent.addHook(BeforeModelCallEvent, () => {
      modelCalls += 1
    })
    // Bounded so a regression fails on the pass count instead of looping forever.
    agent.addHook(AfterInvocationEvent, (event) => {
      if (passes < 5) event.resume = 'try again'
    })

    const result = await agent.invoke('go', { limits: { turns: 1 } })

    expect(result.stopReason).toBe('limitTurns')
    expect(passes).toBe(1)
    expect(modelCalls).toBe(1)
  })
})
