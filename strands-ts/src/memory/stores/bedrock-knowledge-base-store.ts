import {
  BedrockAgentRuntimeClient,
  type BedrockAgentRuntimeClientConfig,
  RetrieveCommand,
  type RetrievalFilter,
} from '@aws-sdk/client-bedrock-agent-runtime'
import {
  BedrockAgentClient,
  type BedrockAgentClientConfig,
  type KnowledgeBaseDocument,
  IngestKnowledgeBaseDocumentsCommand,
} from '@aws-sdk/client-bedrock-agent'
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3'
import { v7 as uuidv7 } from 'uuid'

import type { MemoryEntry, MemoryStore, MemoryStoreConfig, SearchOptions } from '../types.js'
import type { JSONValue } from '../../types/json.js'

/** A typed metadata attribute value, mirroring Bedrock's `MetadataAttributeValue`. */
type AttributeValue =
  | { type: 'STRING'; stringValue: string }
  | { type: 'NUMBER'; numberValue: number }
  | { type: 'BOOLEAN'; booleanValue: boolean }
  | { type: 'STRING_LIST'; stringListValue: string[] }

/** An inline metadata attribute on a `CUSTOM` document, mirroring Bedrock's `MetadataAttribute`. */
type InlineAttribute = { key: string; value: AttributeValue }

/**
 * An attribute entry in an S3 `.metadata.json` sidecar. `includeForEmbedding` is `false` so the
 * attribute is stored for filtering only and does not influence the embedding (matching how inline
 * attributes behave for `CUSTOM` documents).
 */
type SidecarAttribute = { value: AttributeValue; includeForEmbedding: false }

/** Converts a caller metadata value into a Bedrock attribute value, or `undefined` if unsupported. */
function toAttributeValue(value: JSONValue): AttributeValue | undefined {
  if (typeof value === 'string') return { type: 'STRING', stringValue: value }
  if (typeof value === 'number') return { type: 'NUMBER', numberValue: value }
  if (typeof value === 'boolean') return { type: 'BOOLEAN', booleanValue: value }
  if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
    return { type: 'STRING_LIST', stringListValue: value }
  }
  return undefined
}

/**
 * S3 ingestion settings, required when `dataSourceType` is `'S3'`.
 *
 * An S3 data source indexes objects from a bucket — there is no inline-text path — so `add` uploads
 * its content here as an object and then ingests that object. Use the bucket the data source reads
 * from (and a `prefix` within its inclusion prefixes) so the uploaded object stays in sync with the
 * knowledge base across future data-source syncs.
 */
export interface BedrockKnowledgeBaseS3Config {
  /** Bucket to upload content (and metadata sidecars) to before ingestion. */
  bucket: string
  /** Client used to upload objects. The caller owns its construction and credentials. */
  client: S3Client
  /** Key prefix for uploaded objects (e.g. `'memories/'`). A trailing slash is added when missing. */
  prefix: string
}

export interface BedrockKnowledgeBaseStoreConfig extends MemoryStoreConfig {
  knowledgeBaseId: string
  /**
   * The type of data source backing this knowledge base, matching Bedrock's `dataSourceType`. Only
   * `'CUSTOM'` and `'S3'` data sources accept direct document ingestion
   * (`IngestKnowledgeBaseDocuments`), so only those can be written to:
   * - `'CUSTOM'`: `add` ingests its `content` argument as inline text, with scope/metadata attached
   *   as inline attributes.
   * - `'S3'`: `add` uploads its `content` to the configured `s3` bucket and ingests that object, so
   *   the write is self-contained (no separate upload or sync needed). Scope/metadata are written
   *   alongside as a `.metadata.json` sidecar. Requires `s3`.
   * - `'OTHER'`: any other backend (Confluence, SharePoint, Salesforce, Web, SQL/Redshift, …),
   *   which sync from an external store or are query-only and so are read-only.
   *
   * Effective writability is `writable && (dataSourceType === 'CUSTOM' || dataSourceType === 'S3')`:
   * a store is only writable when the caller opts in *and* the backend supports direct ingestion.
   * When omitted, the store is read-only.
   */
  dataSourceType?: 'CUSTOM' | 'S3' | 'OTHER'
  /**
   * Data source to ingest into when writing. Required for `add` to succeed — without it, write
   * calls throw, since the knowledge base has no destination to ingest into.
   */
  dataSourceId?: string
  /** S3 ingestion settings. Required when `dataSourceType` is `'S3'`; ignored otherwise. */
  s3?: BedrockKnowledgeBaseS3Config
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
  private readonly _s3Config: BedrockKnowledgeBaseS3Config | undefined
  private readonly _knowledgeBaseId: string
  private readonly _dataSourceType: 'CUSTOM' | 'S3' | 'OTHER' | undefined
  private readonly _dataSourceId: string | undefined
  private readonly _scope: string | undefined
  private readonly _scopeMetadataKey: string
  private readonly _filter: RetrievalFilter | undefined

  constructor(config: BedrockKnowledgeBaseStoreConfig) {
    this.name = config.name
    if (config.description !== undefined) this.description = config.description
    if (config.maxSearchResults !== undefined) this.maxSearchResults = config.maxSearchResults
    // Only `CUSTOM` and `S3` data sources accept direct ingestion; reject `writable` on any other.
    this.writable = config.writable ?? false
    if (this.writable && config.dataSourceType !== 'CUSTOM' && config.dataSourceType !== 'S3') {
      throw new Error(
        `BedrockKnowledgeBaseStore: writable is true but dataSourceType is '${config.dataSourceType ?? 'undefined'}'. ` +
          "Only 'CUSTOM' and 'S3' data sources support document ingestion; 'OTHER' backends are read-only."
      )
    }

    this._runtimeClient = config.runtimeClient ?? new BedrockAgentRuntimeClient(config.runtimeClientConfig ?? {})
    this._agentClient = config.agentClient
    this._agentClientConfig = config.agentClientConfig
    this._s3Config = config.s3
    this._knowledgeBaseId = config.knowledgeBaseId
    this._dataSourceType = config.dataSourceType
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

    // S3 and CUSTOM data sources accept fundamentally different documents. S3 ingests objects, so
    // its document references objects uploaded to S3 first; CUSTOM ingests the text inline.
    let document: KnowledgeBaseDocument
    if (this._dataSourceType === 'S3') {
      const objectUris = await this._uploadS3Objects(content, metadata)
      document = this._buildS3Document(objectUris)
    } else {
      document = this._buildCustomDocument(content, metadata)
    }

    await this._getAgentClient().send(
      new IngestKnowledgeBaseDocumentsCommand({
        knowledgeBaseId: this._knowledgeBaseId,
        dataSourceId,
        documents: [document],
      })
    )
  }

  /**
   * Uploads the content (and, when there's scope/metadata, a `.metadata.json` sidecar beside it) to
   * S3, returning the `s3://` URIs to reference for ingestion. Bedrock reads these objects and
   * indexes them — scope/metadata can't be sent inline for S3, hence the sidecar convention.
   */
  private async _uploadS3Objects(
    content: string,
    metadata?: Record<string, JSONValue>
  ): Promise<{ contentUri: string; sidecarUri?: string }> {
    const s3 = this._requireS3Config()
    const prefix = s3.prefix.endsWith('/') ? s3.prefix : `${s3.prefix}/`
    const key = `${prefix}${uuidv7()}.txt`

    const contentUri = await this._putObject(s3, key, content, 'text/plain; charset=utf-8')

    const attributes = this._buildSidecarAttributes(metadata)
    if (Object.keys(attributes).length === 0) {
      return { contentUri }
    }

    // The sidecar must sit beside the source object and be named `<object-key>.metadata.json`.
    const sidecar = JSON.stringify({ metadataAttributes: attributes })
    const sidecarUri = await this._putObject(s3, `${key}.metadata.json`, sidecar, 'application/json')
    return { contentUri, sidecarUri }
  }

  /** Uploads a single object to the configured bucket and returns its `s3://` URI. */
  private async _putObject(
    s3: BedrockKnowledgeBaseS3Config,
    key: string,
    body: string,
    contentType: string
  ): Promise<string> {
    await s3.client.send(new PutObjectCommand({ Bucket: s3.bucket, Key: key, Body: body, ContentType: contentType }))
    return `s3://${s3.bucket}/${key}`
  }

  /** Builds a document for an `S3` data source from the uploaded object (and sidecar) URIs. */
  private _buildS3Document({
    contentUri,
    sidecarUri,
  }: {
    contentUri: string
    sidecarUri?: string
  }): KnowledgeBaseDocument {
    const document: KnowledgeBaseDocument = {
      content: {
        dataSourceType: 'S3',
        s3: { s3Location: { uri: contentUri } },
      },
    }

    if (sidecarUri) {
      document.metadata = {
        type: 'S3_LOCATION',
        s3Location: { uri: sidecarUri },
      }
    }

    return document
  }

  /**
   * Builds a document for a `CUSTOM` data source: the text ingested inline, with the scope and any
   * caller metadata attached as inline attributes for retrieval filtering.
   */
  private _buildCustomDocument(content: string, metadata?: Record<string, JSONValue>): KnowledgeBaseDocument {
    const inlineAttributes: InlineAttribute[] = []

    if (this._scope) {
      inlineAttributes.push({ key: this._scopeMetadataKey, value: { type: 'STRING', stringValue: this._scope } })
    }

    if (metadata) {
      for (const [key, value] of Object.entries(metadata)) {
        const attributeValue = toAttributeValue(value)
        if (attributeValue) inlineAttributes.push({ key, value: attributeValue })
      }
    }

    return {
      content: {
        dataSourceType: 'CUSTOM',
        custom: {
          customDocumentIdentifier: { id: uuidv7() },
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
    }
  }

  /** Builds the `metadataAttributes` map for an S3 sidecar from the scope and caller metadata. */
  private _buildSidecarAttributes(metadata?: Record<string, JSONValue>): Record<string, SidecarAttribute> {
    const attributes: Record<string, SidecarAttribute> = {}

    if (this._scope) {
      attributes[this._scopeMetadataKey] = {
        value: { type: 'STRING', stringValue: this._scope },
        includeForEmbedding: false,
      }
    }

    if (metadata) {
      for (const [key, value] of Object.entries(metadata)) {
        const attributeValue = toAttributeValue(value)
        if (attributeValue) attributes[key] = { value: attributeValue, includeForEmbedding: false }
      }
    }

    return attributes
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

  private _requireS3Config(): BedrockKnowledgeBaseS3Config {
    if (!this._s3Config) {
      throw new Error(
        "BedrockKnowledgeBaseStore: s3 config is required when dataSourceType is 'S3'. " +
          'Provide it in the config to enable add().'
      )
    }
    return this._s3Config
  }

  private _getAgentClient(): BedrockAgentClient {
    if (!this._agentClient) {
      this._agentClient = new BedrockAgentClient(this._agentClientConfig ?? {})
    }
    return this._agentClient
  }
}
