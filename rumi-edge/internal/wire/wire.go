// Package wire holds the A2A + MCP JSON-RPC wire types, translated from
// nodeai's protocol layer and stripped of account/world/widget coupling —
// rumi identity is just (agentType, agentId).
package wire

import "encoding/json"

// ---- JSON-RPC 2.0 envelope ----

type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

type Response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *Error          `json:"error,omitempty"`
}

// Error codes — JSON-RPC standard + A2A app-specific.
const (
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603

	TaskNotFound      = -32001
	TaskNotCancelable = -32002
)

// Method names.
const (
	MethodMessageSend     = "message/send"
	MethodMessageStream   = "message/stream"
	MethodTasksGet        = "tasks/get"
	MethodTasksCancel     = "tasks/cancel"
	MethodToolsList       = "tools/list"
	MethodToolsCall       = "tools/call"
	MethodThreadsList     = "threads/list"
	MethodThreadsCreate   = "threads/create"
	MethodMessagesHistory = "messages/history"
	MethodResourcesList   = "resources/list"
	MethodResourcesRead   = "resources/read"
)

// ---- A2A types ----

type Part struct {
	Type     string         `json:"type"` // "text" | "data" | "file"
	Text     string         `json:"text,omitempty"`
	Data     any            `json:"data,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type Message struct {
	Role     string         `json:"role"` // "user" | "agent"
	Parts    []Part         `json:"parts"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type TaskStatus struct {
	State     string   `json:"state"` // submitted|working|completed|failed|canceled
	Message   *Message `json:"message,omitempty"`
	Timestamp string   `json:"timestamp,omitempty"`
}

type Task struct {
	ID        string         `json:"id"`
	ContextID string         `json:"contextId,omitempty"`
	Status    TaskStatus     `json:"status"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// ---- A2A threads ----
//
// rumi has no native thread concept: a "thread" IS an (agentType, agentId), so
// ThreadID == the contextId. threads/list enumerates the agentIds that have
// on-disk history under one agentType; threads/create echoes a fresh contextId
// (the thread materializes on its first message/send — no metadata persists, so
// CreatedAt/UpdatedAt are empty until then).
type Thread struct {
	ThreadID            string `json:"threadId"`
	Title               string `json:"title"`
	CreatedAt           string `json:"createdAt,omitempty"`
	UpdatedAt           string `json:"updatedAt,omitempty"`
	MessageCount        int    `json:"messageCount"`
	FirstMessagePreview string `json:"firstMessagePreview"`
}

// ---- A2A streaming events (message/stream SSE) ----
//
// A2A streaming is task-STATUS + ARTIFACT events, not token deltas. The daemon
// re-emits each worker progress frame as a TaskStatusUpdateEvent and closes the
// stream on a terminal event with Final=true. `Final` is intentionally NOT
// omitempty: clients key on `"final":false` vs `"final":true`, so it is always
// present on the wire (matches nodeai).

type TaskStatusUpdateEvent struct {
	Type      string         `json:"type"` // "status-update"
	TaskID    string         `json:"taskId"`
	ContextID string         `json:"contextId,omitempty"`
	Status    TaskStatus     `json:"status"`
	Final     bool           `json:"final"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// A2AArtifact is agent-produced data (nodeai maps box_updated → artifact-update).
// rumi has no box-write event on the seam yet, so artifact-update is not emitted
// in this version — the type is kept so the streaming wire contract is a faithful
// superset of A2A rather than a strict subset.
type A2AArtifact struct {
	ArtifactID  string         `json:"artifactId,omitempty"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parts       []Part         `json:"parts"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type TaskArtifactUpdateEvent struct {
	Type      string         `json:"type"` // "artifact-update"
	TaskID    string         `json:"taskId"`
	ContextID string         `json:"contextId,omitempty"`
	Artifact  A2AArtifact    `json:"artifact"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// ---- MCP types ----

type MCPToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"inputSchema"`
}

type MCPContentBlock struct {
	Type string `json:"type"` // "text"
	Text string `json:"text,omitempty"`
}

type MCPCallToolResult struct {
	Content []MCPContentBlock `json:"content"`
	IsError bool              `json:"isError"`
}

// MCPResource / MCPResourceContent — an agent's inscribed box keys served as MCP
// resources (NOT its history). URI scheme is rumi://box/{key}; content is always
// JSON text (no blob variant — rumi box values are JSON).
type MCPResource struct {
	URI         string `json:"uri"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

type MCPResourceContent struct {
	URI      string `json:"uri"`
	Text     string `json:"text,omitempty"`
	MimeType string `json:"mimeType"`
}

// ---- Agent card ----

type AgentCard struct {
	Name               string            `json:"name"`
	Description        string            `json:"description"`
	URL                string            `json:"url"`
	Version            string            `json:"version"`
	ProtocolVersion    string            `json:"protocolVersion"`
	Capabilities       map[string]any    `json:"capabilities"`
	Skills             []any             `json:"skills"`
	DefaultInputModes  []string          `json:"defaultInputModes"`
	DefaultOutputModes []string          `json:"defaultOutputModes"`
	Provider           map[string]string `json:"provider,omitempty"`
}
