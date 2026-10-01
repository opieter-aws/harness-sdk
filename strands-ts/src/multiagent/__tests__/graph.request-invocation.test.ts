import { describe, expect, it } from 'vitest'
import { Agent } from '../../agent/agent.js'
import { readInvocation, type Invocation } from '../../agent/invocation.js'
import { AfterInvocationEvent } from '../../hooks/events.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { Graph } from '../graph.js'

describe('Graph request-wide invocation', () => {
  it('folds every node model usage and turn into one shared request total', async () => {
    const agentA = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'A done' },
        { usage: { inputTokens: 10, outputTokens: 25, totalTokens: 35 } }
      ),
      printer: false,
      id: 'a',
    })
    const agentB = new Agent({
      model: new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'B done' },
        { usage: { inputTokens: 20, outputTokens: 35, totalTokens: 55 } }
      ),
      printer: false,
      id: 'b',
    })

    // Capture the shared Invocation from the last node's after-invocation hook,
    // where it is linked to the event.
    let shared: Invocation | undefined
    agentB.addHook(AfterInvocationEvent, (event) => {
      shared = readInvocation(event)
    })

    const graph = new Graph({
      nodes: [agentA, agentB],
      edges: [{ source: 'a', target: 'b' }],
    })

    await graph.invoke('hello')

    // Both nodes accumulate into the single shared Invocation, so usage is the
    // exact sum of both nodes' model usage and turns count both.
    expect(shared?.usage).toEqual({ inputTokens: 30, outputTokens: 60, totalTokens: 90 })
    expect(shared?.turns).toBe(2)
  })
})
