// Package server is the rumi-edge HTTP front: it parses A2A + MCP JSON-RPC,
// translates to/from the Python worker over the seam, and serves the agent
// card. v1 is unary — message/send blocks and returns a completed task.
package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"rumi-edge/internal/wire"
	"rumi-edge/internal/worker"
)

type Server struct {
	w   *worker.Client
	reg *TaskRegistry
}

func New(w *worker.Client) *Server { return &Server{w: w, reg: NewTaskRegistry()} }

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /a2a", s.handleA2A)
	mux.HandleFunc("GET /.well-known/agent.json", s.handleCard)
	return mux
}

func (s *Server) handleCard(w http.ResponseWriter, r *http.Request) {
	card := wire.AgentCard{
		Name:            "rumi",
		Description:     "A slim agentic library — A2A + MCP edge.",
		URL:             "http://" + r.Host + "/a2a",
		Version:         "0.2.0",
		ProtocolVersion: "0.3",
		Capabilities:    map[string]any{"streaming": true, "pushNotifications": false},
		Skills:          []any{},
		DefaultInputModes:  []string{"text", "data"},
		DefaultOutputModes: []string{"text", "data"},
		Provider:           map[string]string{"organization": "rumi"},
	}
	writeJSON(w, http.StatusOK, card)
}

func (s *Server) handleA2A(w http.ResponseWriter, r *http.Request) {
	var req wire.Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeRPCError(w, nil, wire.ParseError, "parse error")
		return
	}
	switch req.Method {
	case wire.MethodMessageSend:
		s.messageSend(w, req)
	case wire.MethodMessageStream:
		s.messageStream(w, r, req)
	case wire.MethodTasksGet:
		s.tasksGet(w, req)
	case wire.MethodTasksCancel:
		s.tasksCancel(w, req)
	case wire.MethodToolsList:
		s.toolsList(w, req)
	case wire.MethodToolsCall:
		s.toolsCall(w, req)
	case wire.MethodThreadsList:
		s.threadsList(w, req)
	case wire.MethodThreadsCreate:
		s.threadsCreate(w, req)
	case wire.MethodMessagesHistory:
		s.messagesHistory(w, req)
	case wire.MethodResourcesList:
		s.resourcesList(w, req)
	case wire.MethodResourcesRead:
		s.resourcesRead(w, req)
	default:
		writeRPCError(w, req.ID, wire.MethodNotFound, "method not found: "+req.Method)
	}
}

// ---- identity ----

type identity struct {
	ContextID string `json:"contextId"`
	AgentType string `json:"agentType"`
	AgentID   string `json:"agentId"`
}

func (id identity) resolve() (agentType, agentID string, err error) {
	if id.ContextID != "" {
		parts := strings.SplitN(id.ContextID, "/", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return "", "", fmt.Errorf("contextId must be 'agentType/agentId'")
		}
		return parts[0], parts[1], nil
	}
	if id.AgentType == "" || id.AgentID == "" {
		return "", "", fmt.Errorf("need contextId or agentType+agentId")
	}
	return id.AgentType, id.AgentID, nil
}

// resolveAgentType extracts just the agentType — from the AgentType field or
// the type-part of a contextId — for the agentType-scoped methods (tools/list,
// threads/list, threads/create) that don't need an agentId.
func (id identity) resolveAgentType() (string, error) {
	if id.AgentType != "" {
		return id.AgentType, nil
	}
	if id.ContextID != "" {
		if t := strings.SplitN(id.ContextID, "/", 2)[0]; t != "" {
			return t, nil
		}
	}
	return "", fmt.Errorf("need agentType")
}

// ---- shared message-param parsing ----

type sendParams struct {
	agentType string
	agentID   string
	text      string
}

// parseMessageParams resolves identity + flattens the message text for both
// message/send and message/stream. On any failure it writes a JSON-RPC error
// and returns ok=false (callers must return without writing anything else).
func (s *Server) parseMessageParams(w http.ResponseWriter, req wire.Request) (sendParams, bool) {
	var p struct {
		identity
		Message wire.Message `json:"message"`
	}
	if err := json.Unmarshal(req.Params, &p); err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, "invalid params: "+err.Error())
		return sendParams{}, false
	}
	agentType, agentID, err := p.resolve()
	if err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return sendParams{}, false
	}
	text := flattenText(p.Message)
	if strings.TrimSpace(text) == "" {
		writeRPCError(w, req.ID, wire.InvalidParams, "message has no text content")
		return sendParams{}, false
	}
	return sendParams{agentType: agentType, agentID: agentID, text: text}, true
}

// ---- message/send (async) ----

// messageSend registers a task, kicks off the run in the background, and returns
// a "submitted" Task immediately. The caller observes progress via tasks/get or
// message/stream and can stop it via tasks/cancel.
func (s *Server) messageSend(w http.ResponseWriter, req wire.Request) {
	p, ok := s.parseMessageParams(w, req)
	if !ok {
		return
	}
	taskID := newID()
	ctxID := p.agentType + "/" + p.agentID
	s.reg.Submit(taskID, ctxID, func() { _ = s.w.Cancel(taskID) })
	go s.runTask(taskID, p.agentType, p.agentID, p.text)
	writeRPCResult(w, req.ID, wire.Task{
		ID: taskID, ContextID: ctxID,
		Status:   wire.TaskStatus{State: "submitted", Timestamp: nowISO()},
		Metadata: map[string]any{"taskId": taskID},
	})
}

// runTask drives one background run, updating the registry from the worker's
// frames. The terminal frame decides the final state. InvokeStream always
// returns a terminal (it synthesizes an error on a broken seam), so this never
// hangs.
func (s *Server) runTask(taskID, agentType, agentID, text string) {
	term, _ := s.w.InvokeStream(taskID, agentType, agentID, text, func(f worker.Frame) {
		if !f.Terminal() {
			s.reg.SetState(taskID, "working")
		}
	})
	switch term.Type {
	case "final":
		s.reg.SetCompleted(taskID, stringify(term.Content), term.Metadata)
	case "canceled":
		s.reg.SetCanceled(taskID)
	default: // "error" or anything unexpected
		s.reg.SetFailed(taskID, term.Message)
	}
}

// ---- message/stream (inline SSE on POST /a2a) ----

// messageStream responds with text/event-stream directly: each worker progress
// frame becomes a TaskStatusUpdateEvent(working), and a terminal frame becomes a
// final event (Final=true) carrying the answer. A2A streaming is status/artifact
// events, not token deltas. A client disconnect cancels the underlying run.
func (s *Server) messageStream(w http.ResponseWriter, r *http.Request, req wire.Request) {
	p, ok := s.parseMessageParams(w, req)
	if !ok {
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeRPCError(w, req.ID, wire.InternalError, "streaming unsupported (no http.Flusher)")
		return
	}
	taskID := newID()
	ctxID := p.agentType + "/" + p.agentID
	s.reg.Submit(taskID, ctxID, func() { _ = s.w.Cancel(taskID) })

	// Client disconnect → cancel the worker run. The watcher self-terminates
	// when the handler returns (net/http cancels the request context then too);
	// the resulting Cancel of a finished task is a harmless no-op.
	go func() {
		<-r.Context().Done()
		_ = s.w.Cancel(taskID)
	}()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	writeSSE := func(ev any) {
		b, _ := json.Marshal(ev)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	term, _ := s.w.InvokeStream(taskID, p.agentType, p.agentID, p.text, func(f worker.Frame) {
		if f.Terminal() {
			return
		}
		s.reg.SetState(taskID, "working")
		writeSSE(wire.TaskStatusUpdateEvent{
			Type: "status-update", TaskID: taskID, ContextID: ctxID, Final: false,
			Status: wire.TaskStatus{State: "working", Message: stageMessage(f), Timestamp: nowISO()},
		})
	})

	final := wire.TaskStatusUpdateEvent{
		Type: "status-update", TaskID: taskID, ContextID: ctxID, Final: true,
	}
	switch term.Type {
	case "final":
		answer := stringify(term.Content)
		s.reg.SetCompleted(taskID, answer, term.Metadata)
		final.Status = wire.TaskStatus{
			State:     "completed",
			Message:   &wire.Message{Role: "agent", Parts: []wire.Part{{Type: "text", Text: answer}}},
			Timestamp: nowISO(),
		}
		final.Metadata = term.Metadata
	case "canceled":
		s.reg.SetCanceled(taskID)
		final.Status = wire.TaskStatus{State: "canceled", Timestamp: nowISO()}
	default: // "error" or anything unexpected
		s.reg.SetFailed(taskID, term.Message)
		final.Status = wire.TaskStatus{
			State:     "failed",
			Message:   &wire.Message{Role: "agent", Parts: []wire.Part{{Type: "text", Text: term.Message}}},
			Timestamp: nowISO(),
		}
	}
	writeSSE(final)
}

// stageMessage renders a progress frame as an A2A message carrying a data part.
// rumi-aware clients read `__rumi_stage`; generic A2A clients see a working
// status-update with an opaque data part they can ignore.
func stageMessage(f worker.Frame) *wire.Message {
	data := map[string]any{"__rumi_stage": f.Stage}
	if f.Model != "" {
		data["model"] = f.Model
	}
	if len(f.Tools) > 0 {
		data["tools"] = f.Tools
	}
	if f.CostUSD != nil {
		data["cost_usd"] = *f.CostUSD
	}
	return &wire.Message{Role: "agent", Parts: []wire.Part{{Type: "data", Data: data}}}
}

// ---- tasks/get + tasks/cancel ----

func parseTaskID(req wire.Request) string {
	var p struct {
		ID     string `json:"id"`
		TaskID string `json:"taskId"`
	}
	_ = json.Unmarshal(req.Params, &p)
	if p.ID != "" {
		return p.ID
	}
	return p.TaskID
}

func (s *Server) tasksGet(w http.ResponseWriter, req wire.Request) {
	id := parseTaskID(req)
	if id == "" {
		writeRPCError(w, req.ID, wire.InvalidParams, "missing task id ('id' or 'taskId')")
		return
	}
	rec, ok := s.reg.Get(id)
	if !ok {
		writeRPCError(w, req.ID, wire.TaskNotFound, "task not found: "+id)
		return
	}
	task := wire.Task{
		ID: id, ContextID: rec.ContextID,
		Status:   wire.TaskStatus{State: rec.State, Timestamp: rec.Updated},
		Metadata: rec.Metadata,
	}
	if rec.Answer != "" && (rec.State == "completed" || rec.State == "failed") {
		task.Status.Message = &wire.Message{Role: "agent", Parts: []wire.Part{{Type: "text", Text: rec.Answer}}}
	}
	writeRPCResult(w, req.ID, task)
}

func (s *Server) tasksCancel(w http.ResponseWriter, req wire.Request) {
	id := parseTaskID(req)
	if id == "" {
		writeRPCError(w, req.ID, wire.InvalidParams, "missing task id ('id' or 'taskId')")
		return
	}
	cancelable, found := s.reg.Cancel(id)
	if !found {
		writeRPCError(w, req.ID, wire.TaskNotFound, "task not found: "+id)
		return
	}
	if !cancelable {
		writeRPCError(w, req.ID, wire.TaskNotCancelable, "task already finished: "+id)
		return
	}
	// reg.Cancel fired the cooperative worker cancel; the worker's terminal
	// "canceled" frame drives the authoritative state. Return optimistically.
	ctxID := ""
	if rec, ok := s.reg.Get(id); ok {
		ctxID = rec.ContextID
	}
	writeRPCResult(w, req.ID, wire.Task{
		ID: id, ContextID: ctxID,
		Status: wire.TaskStatus{State: "canceled", Timestamp: nowISO()},
	})
}

// ---- tools/list ----

func (s *Server) toolsList(w http.ResponseWriter, req wire.Request) {
	var id identity
	_ = json.Unmarshal(req.Params, &id) // agentType (and optional id) only
	agentType, err := id.resolveAgentType()
	if err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return
	}
	entries, err := s.w.ListTools(agentType)
	if err != nil {
		writeRPCError(w, req.ID, wire.InternalError, err.Error())
		return
	}
	defs := make([]wire.MCPToolDef, 0, len(entries))
	for _, e := range entries {
		defs = append(defs, toMCPToolDef(e))
	}
	writeRPCResult(w, req.ID, map[string]any{"tools": defs})
}

// ---- tools/call ----

func (s *Server) toolsCall(w http.ResponseWriter, req wire.Request) {
	var p struct {
		identity
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	if err := json.Unmarshal(req.Params, &p); err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, "invalid params: "+err.Error())
		return
	}
	agentType, agentID, err := p.resolve()
	if err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return
	}
	if p.Name == "" {
		writeRPCError(w, req.ID, wire.InvalidParams, "need tool name")
		return
	}
	result, err := s.w.CallTool(agentType, agentID, p.Name, p.Arguments)
	if err != nil {
		// MCP convention: a tool failure is a result with isError, not a JSON-RPC error.
		writeRPCResult(w, req.ID, wire.MCPCallToolResult{
			Content: []wire.MCPContentBlock{{Type: "text", Text: err.Error()}},
			IsError: true,
		})
		return
	}
	writeRPCResult(w, req.ID, wire.MCPCallToolResult{
		Content: []wire.MCPContentBlock{{Type: "text", Text: stringify(result)}},
		IsError: false,
	})
}

// ---- threads/list + threads/create ----
//
// A "thread" is an (agentType, agentId); threadId == contextId. threads/list is
// scoped by agentType (enumerates the agentIds with on-disk history). The worker
// owns the enumeration (it owns the World / data dir).

func (s *Server) threadsList(w http.ResponseWriter, req wire.Request) {
	var id identity
	_ = json.Unmarshal(req.Params, &id) // agentType, or the type-part of contextId
	agentType, err := id.resolveAgentType()
	if err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return
	}
	threads, err := s.w.ListThreads(agentType)
	if err != nil {
		writeWorkerErr(w, req, err)
		return
	}
	writeRPCResult(w, req.ID, map[string]any{"threads": threads})
}

// threadsCreate is handled entirely Go-side: a created-but-unused thread has no
// on-disk state in rumi, so "create" just mints/validates an agentId and echoes
// the contextId. The thread materializes (becomes listable) on its first
// message/send. No worker round-trip, no persistence.
func (s *Server) threadsCreate(w http.ResponseWriter, req wire.Request) {
	var p struct {
		identity
		AgentID string `json:"agentId"`
		Title   string `json:"title"`
	}
	if err := json.Unmarshal(req.Params, &p); err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, "invalid params: "+err.Error())
		return
	}
	agentType, err := p.resolveAgentType()
	if err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return
	}
	agentID := p.AgentID
	if agentID == "" {
		agentID = newID()
	} else if err := validateAgentID(agentID); err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return
	}
	title := p.Title
	if title == "" {
		title = agentID
	}
	now := nowISO()
	writeRPCResult(w, req.ID, wire.Thread{
		ThreadID:  agentType + "/" + agentID,
		Title:     title,
		CreatedAt: now,
		UpdatedAt: now,
	})
}

// validateAgentID mirrors rumi's Agent.__init__ id rules so a minted/echoed
// threadId is one the worker will actually accept on the first message/send.
func validateAgentID(id string) error {
	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("agentId must be non-empty")
	}
	if strings.ContainsAny(id, `/\`) {
		return fmt.Errorf("agentId must not contain '/' or '\\'")
	}
	if id == "." || id == ".." {
		return fmt.Errorf("agentId must not be '.' or '..'")
	}
	return nil
}

// ---- messages/history ----

func (s *Server) messagesHistory(w http.ResponseWriter, req wire.Request) {
	var p struct {
		identity
		Limit  int    `json:"limit"`
		Before string `json:"before"`
	}
	if err := json.Unmarshal(req.Params, &p); err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, "invalid params: "+err.Error())
		return
	}
	agentType, agentID, err := p.resolve()
	if err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return
	}
	res, err := s.w.History(agentType, agentID, p.Limit, p.Before)
	if err != nil {
		writeWorkerErr(w, req, err)
		return
	}
	out := map[string]any{"messages": res.Messages, "hasMore": res.HasMore}
	if res.NextCursor != "" {
		out["nextCursor"] = res.NextCursor
	}
	writeRPCResult(w, req.ID, out)
}

// ---- resources/list + resources/read ----

func (s *Server) resourcesList(w http.ResponseWriter, req wire.Request) {
	var id identity
	if err := json.Unmarshal(req.Params, &id); err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, "invalid params: "+err.Error())
		return
	}
	agentType, agentID, err := id.resolve()
	if err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return
	}
	resources, err := s.w.ListResources(agentType, agentID)
	if err != nil {
		writeWorkerErr(w, req, err)
		return
	}
	writeRPCResult(w, req.ID, map[string]any{"resources": resources})
}

func (s *Server) resourcesRead(w http.ResponseWriter, req wire.Request) {
	var p struct {
		identity
		URI string `json:"uri"`
	}
	if err := json.Unmarshal(req.Params, &p); err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, "invalid params: "+err.Error())
		return
	}
	agentType, agentID, err := p.resolve()
	if err != nil {
		writeRPCError(w, req.ID, wire.InvalidParams, err.Error())
		return
	}
	if p.URI == "" {
		writeRPCError(w, req.ID, wire.InvalidParams, "need uri")
		return
	}
	contents, err := s.w.ReadResource(agentType, agentID, p.URI)
	if err != nil {
		writeWorkerErr(w, req, err)
		return
	}
	writeRPCResult(w, req.ID, map[string]any{"contents": contents})
}

// ---- helpers ----

// writeWorkerErr surfaces a worker JSON-RPC error with its original code (e.g.
// -32602 for an unknown agent type / resource / cursor), falling back to
// InternalError for transport-level failures.
func writeWorkerErr(w http.ResponseWriter, req wire.Request, err error) {
	var we *worker.WorkerError
	if errors.As(err, &we) {
		writeRPCError(w, req.ID, we.Code, we.Message)
		return
	}
	writeRPCError(w, req.ID, wire.InternalError, err.Error())
}

// toMCPToolDef translates an OpenAI function-tool dict
// ({type:function, function:{name,description,parameters}}) into an MCPToolDef.
func toMCPToolDef(e worker.ToolEntry) wire.MCPToolDef {
	def := wire.MCPToolDef{Name: e.Name, InputSchema: map[string]any{}}
	if fn, ok := e.Schema["function"].(map[string]any); ok {
		if n, ok := fn["name"].(string); ok && n != "" {
			def.Name = n
		}
		if d, ok := fn["description"].(string); ok {
			def.Description = d
		}
		if params, ok := fn["parameters"].(map[string]any); ok {
			def.InputSchema = params
		}
	}
	return def
}

func flattenText(m wire.Message) string {
	var sb strings.Builder
	for _, p := range m.Parts {
		if p.Type == "text" && p.Text != "" {
			if sb.Len() > 0 {
				sb.WriteByte('\n')
			}
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

func stringify(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}

func newID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func nowISO() string { return time.Now().UTC().Format(time.RFC3339) }

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeRPCResult(w http.ResponseWriter, id json.RawMessage, result any) {
	writeJSON(w, http.StatusOK, wire.Response{JSONRPC: "2.0", ID: id, Result: result})
}

func writeRPCError(w http.ResponseWriter, id json.RawMessage, code int, msg string) {
	writeJSON(w, http.StatusOK, wire.Response{JSONRPC: "2.0", ID: id, Error: &wire.Error{Code: code, Message: msg}})
}
