// Package worker is the client for the Go↔Python seam: it speaks newline-
// delimited JSON-RPC to the rumi Python worker over loopback TCP. The worker
// runs one whirl() at a time, so calls are serialized by a mutex.
package worker

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"rumi-edge/internal/wire"
)

type Client struct {
	addr string
	mu   sync.Mutex // serialize: one in-flight worker call at a time
}

func New(addr string) *Client { return &Client{addr: addr} }

type rpcReq struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params"`
}

type rpcErr struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type rpcResp struct {
	Result json.RawMessage `json:"result"`
	Error  *rpcErr         `json:"error"`
}

// call dials the worker, sends one JSON-RPC request, reads one reply, and
// unmarshals result into out (if non-nil).
func (c *Client) call(method string, params any, out any) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	conn, err := net.DialTimeout("tcp", c.addr, 5*time.Second)
	if err != nil {
		return fmt.Errorf("dial worker %s: %w", c.addr, err)
	}
	defer conn.Close()

	body, err := json.Marshal(rpcReq{JSONRPC: "2.0", ID: 1, Method: method, Params: params})
	if err != nil {
		return err
	}
	if _, err := conn.Write(append(body, '\n')); err != nil {
		return err
	}

	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		return fmt.Errorf("read worker reply: %w", err)
	}
	var r rpcResp
	if err := json.Unmarshal([]byte(line), &r); err != nil {
		return fmt.Errorf("decode worker reply: %w", err)
	}
	if r.Error != nil {
		return &WorkerError{Code: r.Error.Code, Message: r.Error.Message}
	}
	if out != nil {
		return json.Unmarshal(r.Result, out)
	}
	return nil
}

// WorkerError carries a JSON-RPC error returned by the Python worker.
type WorkerError struct {
	Code    int
	Message string
}

func (e *WorkerError) Error() string { return fmt.Sprintf("worker error %d: %s", e.Code, e.Message) }

// ---- typed calls ----

type InvokeResult struct {
	Idea     map[string]any `json:"idea"`
	Content  any            `json:"content"`
	Metadata map[string]any `json:"metadata"`
}

func (c *Client) Invoke(agentType, agentID string, message string) (*InvokeResult, error) {
	var out InvokeResult
	err := c.call("invoke", map[string]any{
		"agent_type": agentType, "agent_id": agentID, "message": message,
	}, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

type ToolEntry struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"` // OpenAI function-tool dict
}

func (c *Client) ListTools(agentType string) ([]ToolEntry, error) {
	var out struct {
		Tools []ToolEntry `json:"tools"`
	}
	if err := c.call("list_tools", map[string]any{"agent_type": agentType}, &out); err != nil {
		return nil, err
	}
	return out.Tools, nil
}

func (c *Client) CallTool(agentType, agentID, name string, args map[string]any) (any, error) {
	var out struct {
		Result any `json:"result"`
	}
	err := c.call("call_tool", map[string]any{
		"agent_type": agentType, "agent_id": agentID, "name": name, "arguments": args,
	}, &out)
	if err != nil {
		return nil, err
	}
	return out.Result, nil
}

// ---- Phase 2c reads (threads / history / resources) ----
//
// The worker emits final A2A/MCP-shaped camelCase dicts, so these unmarshal
// straight into the wire types and the server passes them through.

func (c *Client) ListThreads(agentType string) ([]wire.Thread, error) {
	var out struct {
		Threads []wire.Thread `json:"threads"`
	}
	if err := c.call("list_threads", map[string]any{"agent_type": agentType}, &out); err != nil {
		return nil, err
	}
	return out.Threads, nil
}

// HistoryResult is the messages/history envelope: a window of A2A messages plus
// an opaque cursor for paging further back.
type HistoryResult struct {
	Messages   []wire.Message `json:"messages"`
	HasMore    bool           `json:"hasMore"`
	NextCursor string         `json:"nextCursor"`
}

func (c *Client) History(agentType, agentID string, limit int, before string) (*HistoryResult, error) {
	params := map[string]any{"agent_type": agentType, "agent_id": agentID}
	if limit > 0 {
		params["limit"] = limit
	}
	if before != "" {
		params["before"] = before
	}
	var out HistoryResult
	if err := c.call("history", params, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) ListResources(agentType, agentID string) ([]wire.MCPResource, error) {
	var out struct {
		Resources []wire.MCPResource `json:"resources"`
	}
	if err := c.call("list_resources", map[string]any{
		"agent_type": agentType, "agent_id": agentID,
	}, &out); err != nil {
		return nil, err
	}
	return out.Resources, nil
}

func (c *Client) ReadResource(agentType, agentID, uri string) ([]wire.MCPResourceContent, error) {
	var out struct {
		Contents []wire.MCPResourceContent `json:"contents"`
	}
	if err := c.call("read_resource", map[string]any{
		"agent_type": agentType, "agent_id": agentID, "uri": uri,
	}, &out); err != nil {
		return nil, err
	}
	return out.Contents, nil
}

// ---- streaming (invoke_stream) ----

// Frame is one neutral progress/terminal event from the worker's invoke_stream.
// Progress frames have Type=="status" (with a Stage); terminal frames have Type
// in {"final","error","canceled"}. The daemon translates these into A2A events.
type Frame struct {
	Type     string         `json:"type"`
	Stage    string         `json:"stage,omitempty"`
	Model    string         `json:"model,omitempty"`
	Tools    []string       `json:"tools,omitempty"`
	CostUSD  *float64       `json:"cost_usd,omitempty"`
	Idea     map[string]any `json:"idea,omitempty"`
	Content  any            `json:"content,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
	Message  string         `json:"message,omitempty"`
}

// Terminal reports whether this frame ends the stream.
func (f Frame) Terminal() bool {
	return f.Type == "final" || f.Type == "error" || f.Type == "canceled"
}

// InvokeStream runs a streaming whirl on the worker. It dials its OWN connection
// (deliberately NOT under c.mu — streams must run concurrently with each other
// and with Cancel), sends one invoke_stream request, and calls onFrame for each
// frame as it arrives. It returns the terminal frame. If the connection ends
// before a terminal frame, it synthesizes an error frame so the caller's run
// loop never hangs.
func (c *Client) InvokeStream(taskID, agentType, agentID, message string, onFrame func(Frame)) (Frame, error) {
	conn, err := net.DialTimeout("tcp", c.addr, 5*time.Second)
	if err != nil {
		return Frame{Type: "error", Message: err.Error()}, fmt.Errorf("dial worker %s: %w", c.addr, err)
	}
	defer conn.Close()

	body, err := json.Marshal(rpcReq{JSONRPC: "2.0", ID: 1, Method: "invoke_stream", Params: map[string]any{
		"task_id": taskID, "agent_type": agentType, "agent_id": agentID, "message": message,
	}})
	if err != nil {
		return Frame{Type: "error", Message: err.Error()}, err
	}
	if _, err := conn.Write(append(body, '\n')); err != nil {
		return Frame{Type: "error", Message: err.Error()}, err
	}

	br := bufio.NewReader(conn)
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			// EOF / read error before a terminal frame: synthesize one so the
			// caller's run loop always terminates.
			return Frame{Type: "error", Message: "stream ended before terminal: " + err.Error()}, nil
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var f Frame
		if json.Unmarshal([]byte(line), &f) != nil {
			continue // skip a malformed line rather than aborting the whole stream
		}
		if onFrame != nil {
			onFrame(f)
		}
		if f.Terminal() {
			return f, nil
		}
	}
}

// Cancel asks the worker to cooperatively stop the running whirl for taskID.
// It uses the one-shot call path on its own short-lived connection, so it works
// even while an InvokeStream is mid-flight.
func (c *Client) Cancel(taskID string) error {
	return c.call("cancel", map[string]any{"task_id": taskID}, nil)
}
