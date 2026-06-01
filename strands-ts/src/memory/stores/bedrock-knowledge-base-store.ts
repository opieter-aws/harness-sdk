import {
  BedrockAgentRuntimeClient,
  type BedrockAgentRuntimeClientConfig,
  RetrieveCommand,
  type RetrievalFilter,
} from '@aws-sdk/client-bedrock-agent-runtime'
import {
  BedrockAgentClient,
  type BedrockAgentClientConfig,
  IngestKnowledgeBaseDocumentsCommand,
} from '@aws-sdk/client-bedrock-agent'
import { v7 as uuidv7 } from 'uuid'

import type { MemoryEntry, MemoryStore, MemoryStoreConfig, SearchOptions } from '../types.js'
import type { JSONValue } from '../../types/json.js'

export interface BedrockKnowledgeBaseStoreConfig extends MemoryStoreConfig {
  knowledgeBaseId: string
  /**
   * The kind of backend this knowledge base is. Only `'custom'` data sources accept direct
   * document ingestion (`IngestKnowledgeBaseDocuments`), so only a `'custom'` store can be written
   * to. `'managed'` sources (S3, Confluence, SharePoint, Salesforce, Web) ingest by syncing from an
   * external store, and `'structured'` stores (SQL/Redshift) are query-only — neither can `add`.
   *
   * Effective writability is `writable && kind === 'custom'`: a store is only writable when the
   * caller opts in *and* the backend supports ingestion. When omitted, the store is read-only.
   */
  kind?: 'custom' | 'managed' | 'structured'
  /**
   * Data source to ingest into when writing. Required for `add` to succeed — without it, write
   * calls throw, since the knowledge base has no destination to ingest into.
   */
  dataSourceId?: string
  scope?: string
  scopeMetadataKey?: string
  filter?: RetrievalFilter
  runtimeClientConfig?: BedrockAgentRuntimeClientConfig
  runtimeClient?: BedrockAgentRuntimeClient
  agentClientConfig?: BedrockAgentClientConfig
  agentClient?: BedrockAgentClient
}

export class BedrockKnowledgeBaseStore implements MemoryStore {
  readonly name: string
  readonly description?: string
  readonly maxSearchResults?: number
  readonly writable: boolean

  private readonly _runtimeClient: BedrockAgentRuntimeClient
  private _agentClient: BedrockAgentClient | undefined
  private readonly _agentClientConfig: BedrockAgentClientConfig | undefined
  private readonly _knowledgeBaseId: string
  private readonly _dataSourceId: string | undefined
  private readonly _scope: string | undefined
  private readonly _scopeMetadataKey: string
  private readonly _filter: RetrievalFilter | undefined

  constructor(config: BedrockKnowledgeBaseStoreConfig) {
    this.name = config.name
    if (config.description !== undefined) this.description = config.description
    if (config.maxSearchResults !== undefined) this.maxSearchResults = config.maxSearchResults
    // Writing requires both caller intent (`writable`) and a backend that can ingest documents.
    // Only a `custom` data source accepts ingestion; `managed` (S3/Confluence/etc.) and `structured`
    // (SQL/Redshift) backends cannot. Asking to write to one of those is a contradiction, so fail
    // fast rather than silently downgrading to read-only.
    this.writable = config.writable ?? false
    if (this.writable && config.kind !== 'custom') {
      throw new Error(
        `BedrockKnowledgeBaseStore: writable is true but kind is '${config.kind ?? 'undefined'}'. ` +
          "Only a 'custom' data source supports document ingestion; 'managed' and 'structured' backends are read-only."
      )
    }

    this._runtimeClient = config.runtimeClient ?? new BedrockAgentRuntimeClient(config.runtimeClientConfig ?? {})
    this._agentClient = config.agentClient
    this._agentClientConfig = config.agentClientConfig
    this._knowledgeBaseId = config.knowledgeBaseId
    this._dataSourceId = config.dataSourceId
    this._scope = config.scope
    this._scopeMetadataKey = config.scopeMetadataKey ?? 'namespace'

    if (config.filter) {
      this._filter = config.filter
    } else if (config.scope) {
      this._filter = {
        equals: {
          key: this._scopeMetadataKey,
          value: config.scope,
        },
      }
    }
  }

  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    const limit = options?.maxSearchResults ?? this.maxSearchResults ?? 10

    const response = await this._runtimeClient.send(
      new RetrieveCommand({
        knowledgeBaseId: this._knowledgeBaseId,
        retrievalQuery: { text: query },
        retrievalConfiguration: {
          vectorSearchConfiguration: {
            numberOfResults: limit,
            ...(this._filter && { filter: this._filter }),
          },
        },
      })
    )

    return (response.retrievalResults ?? []).map((result) => {
      const metadata: Record<string, JSONValue> = {}
      if (result.metadata) {
        for (const [key, value] of Object.entries(result.metadata)) {
          metadata[key] = value as JSONValue
        }
      }
      if (result.location) {
        metadata._location = result.location as unknown as JSONValue
      }
      if (result.score != null) {
        metadata.score = result.score
      }

      return {
        content: result.content?.text ?? '',
        metadata,
      }
    })
  }

  async add(content: string, metadata?: Record<string, JSONValue>): Promise<void> {
    const dataSourceId = this._requireDataSourceId()
    const id = uuidv7()

    const inlineAttributes: Array<{
      key: string
      value:
        | { type: 'STRING'; stringValue: string }
        | { type: 'NUMBER'; numberValue: number }
        | { type: 'BOOLEAN'; booleanValue: boolean }
    }> = []

    if (this._scope) {
      inlineAttributes.push({
        key: this._scopeMetadataKey,
        value: { type: 'STRING' as const, stringValue: this._scope },
      })
    }

    if (metadata) {
      for (const [key, value] of Object.entries(metadata)) {
        if (typeof value === 'string') {
          inlineAttributes.push({
            key,
            value: { type: 'STRING' as const, stringValue: value },
          })
        } else if (typeof value === 'number') {
          inlineAttributes.push({
            key,
            value: { type: 'NUMBER' as const, numberValue: value },
          })
        } else if (typeof value === 'boolean') {
          inlineAttributes.push({
            key,
            value: { type: 'BOOLEAN' as const, booleanValue: value },
          })
        }
      }
    }

    await this._getAgentClient().send(
      new IngestKnowledgeBaseDocumentsCommand({
        knowledgeBaseId: this._knowledgeBaseId,
        dataSourceId,
        documents: [
          {
            content: {
              dataSourceType: 'CUSTOM',
              custom: {
                customDocumentIdentifier: { id },
                sourceType: 'IN_LINE',
                inlineContent: {
                  type: 'TEXT',
                  textContent: { data: content },
                },
              },
            },
            metadata: {
              type: 'IN_LINE_ATTRIBUTE',
              inlineAttributes,
            },
          },
        ],
      })
    )
  }

  private _requireDataSourceId(): string {
    if (!this._dataSourceId) {
      throw new Error(
        'BedrockKnowledgeBaseStore: dataSourceId is required for write operations. ' +
          'Provide it in the config to enable add().'
      )
    }
    return this._dataSourceId
  }

  private _getAgentClient(): BedrockAgentClient {
    if (!this._agentClient) {
      this._agentClient = new BedrockAgentClient(this._agentClientConfig ?? {})
    }
    return this._agentClient
  }
}
