package server

import (
	"bufio"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"rumi-edge/internal/wire"
	"rumi-edge/internal/worker"
)

// fakeWorker stands in for the Python worker: one canned reply per method.
func fakeWorker(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				line, err := bufio.NewReader(c).ReadString('\n')
				if err != nil {
					return
				}
				var req struct {
					Method string `json:"method"`
					Params struct {
						URI string `json:"uri"`
					} `json:"params"`
				}
				_ = json.Unmarshal([]byte(line), &req)
				var resp string
				switch req.Method {
				case "invoke":
					resp = `{"jsonrpc":"2.0","id":1,"result":{"idea":{"kind":"text"},"content":"hello from fake","metadata":{"cost_usd":0.001}}}`
				case "list_tools":
					resp = `{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"enrich_company","schema":{"type":"function","function":{"name":"enrich_company","description":"Enrich a company.","parameters":{"type":"object","properties":{"company_name":{"type":"string"}}}}}}]}}`
				case "call_tool":
					resp = `{"jsonrpc":"2.0","id":1,"result":{"result":{"business":"Northwind"}}}`
				case "list_threads":
					resp = `{"jsonrpc":"2.0","id":1,"result":{"threads":[{"threadId":"Poet/p1","title":"p1","messageCount":2,"createdAt":"2026-05-29T00:00:00Z","updatedAt":"2026-05-29T00:00:01Z","firstMessagePreview":"hello there"}]}}`
				case "history":
					resp = `{"jsonrpc":"2.0","id":1,"result":{"messages":[{"role":"user","parts":[{"type":"text","text":"hello there"}]},{"role":"agent","parts":[{"type":"text","text":"first reply"}]}],"hasMore":false}}`
				case "list_resources":
					resp = `{"jsonrpc":"2.0","id":1,"result":{"resources":[{"uri":"rumi://box/demo.key","name":"demo.key","description":"Box key: demo.key","mimeType":"application/json"}]}}`
				case "read_resource":
					if strings.Contains(req.Params.URI, "nope") {
						resp = `{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"unknown resource: 'nope'"}}`
					} else {
						resp = `{"jsonrpc":"2.0","id":1,"result":{"contents":[{"uri":"rumi://box/demo.key","text":"{\"hello\": \"world\"}","mimeType":"application/json"}]}}`
					}
				default:
					resp = `{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"nope"}}`
				}
				_, _ = c.Write([]byte(resp + "\n"))
			}(conn)
		}
	}()
	return ln.Addr().String()
}

func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(New(worker.New(fakeWorker(t))).Routes())
	t.Cleanup(srv.Close)
	return srv
}

// streamFakeWorker stands in for the Python worker's invoke_stream + cancel.
// invoke_stream emits progress frames then a terminal frame; if the request
// message is "BLOCK" it waits (emitting a working frame first) until a matching
// cancel arrives, then emits a `canceled` terminal — giving cancel tests a
// deterministic window. cancel signals the per-task channel.
func streamFakeWorker(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	var mu sync.Mutex
	cancels := map[string]chan struct{}{}
	signal := func(taskID string) {
		mu.Lock()
		defer mu.Unlock()
		if ch, ok := cancels[taskID]; ok {
			select {
			case <-ch: // already closed
			default:
				close(ch)
			}
		}
	}
	register := func(taskID string) chan struct{} {
		mu.Lock()
		defer mu.Unlock()
		ch := make(chan struct{})
		cancels[taskID] = ch
		return ch
	}

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				line, err := bufio.NewReader(c).ReadString('\n')
				if err != nil {
					return
				}
				var req struct {
					Method string `json:"method"`
					Params struct {
						TaskID  string `json:"task_id"`
						Message string `json:"message"`
					} `json:"params"`
				}
				_ = json.Unmarshal([]byte(line), &req)
				write := func(s string) { _, _ = c.Write([]byte(s + "\n")) }

				switch req.Method {
				case "invoke_stream":
					write(`{"type":"status","stage":"whirl_start"}`)
					write(`{"type":"status","stage":"llm_call","model":"fake"}`)
					if req.Params.Message == "BLOCK" {
						ch := register(req.Params.TaskID)
						select {
						case <-ch:
							write(`{"type":"canceled"}`)
						case <-time.After(5 * time.Second):
							write(`{"type":"final","content":"unblocked","metadata":{}}`)
						}
					} else {
						write(`{"type":"status","stage":"llm_done","cost_usd":0.001}`)
						write(`{"type":"final","content":"hello from fake","metadata":{"cost_usd":0.001}}`)
					}
				case "cancel":
					signal(req.Params.TaskID)
					write(`{"jsonrpc":"2.0","id":1,"result":{"canceled":true,"found":true}}`)
				default:
					write(`{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"nope"}}`)
				}
			}(conn)
		}
	}()
	return ln.Addr().String()
}

func newStreamTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(New(worker.New(streamFakeWorker(t))).Routes())
	t.Cleanup(srv.Close)
	return srv
}

// waitForState polls tasks/get until the task reaches `want` (or fails).
func waitForState(t *testing.T, base, taskID, want string) wire.Task {
	t.Helper()
	for i := 0; i < 200; i++ {
		var g struct {
			Result wire.Task `json:"result"`
		}
		postA2A(t, base, `{"jsonrpc":"2.0","id":9,"method":"tasks/get","params":{"id":"`+taskID+`"}}`, &g)
		if g.Result.Status.State == want {
			return g.Result
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("task %s never reached state %q", taskID, want)
	return wire.Task{}
}

func postA2A(t *testing.T, base, body string, out any) {
	t.Helper()
	resp, err := http.Post(base+"/a2a", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		t.Fatal(err)
	}
}

func TestMessageSendAsyncSubmitted(t *testing.T) {
	srv := newStreamTestServer(t)
	var out struct {
		Result wire.Task   `json:"result"`
		Error  *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"agentType":"Poet","agentId":"p1","message":{"role":"user","parts":[{"type":"text","text":"who is ready?"}]}}}`, &out)
	if out.Error != nil {
		t.Fatalf("unexpected error: %+v", out.Error)
	}
	// message/send is async in 2b: returns submitted immediately.
	if out.Result.Status.State != "submitted" {
		t.Fatalf("state = %q (want submitted)", out.Result.Status.State)
	}
	if out.Result.ContextID != "Poet/p1" {
		t.Fatalf("contextId = %q", out.Result.ContextID)
	}
	// The background run completes; tasks/get eventually reports it.
	done := waitForState(t, srv.URL, out.Result.ID, "completed")
	if got := done.Status.Message.Parts[0].Text; got != "hello from fake" {
		t.Fatalf("answer = %q", got)
	}
}

func TestMessageSendViaContextID(t *testing.T) {
	srv := newStreamTestServer(t)
	var out struct {
		Result wire.Task `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"contextId":"Poet/p1","message":{"role":"user","parts":[{"type":"text","text":"hi"}]}}}`, &out)
	if out.Result.Status.State != "submitted" {
		t.Fatalf("state = %q (want submitted)", out.Result.Status.State)
	}
	waitForState(t, srv.URL, out.Result.ID, "completed")
}

func TestTasksGetUnknownIsNotFound(t *testing.T) {
	srv := newStreamTestServer(t)
	var out struct {
		Error *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"tasks/get","params":{"id":"ghost"}}`, &out)
	if out.Error == nil || out.Error.Code != wire.TaskNotFound {
		t.Fatalf("want TaskNotFound, got %+v", out.Error)
	}
}

func TestMessageStreamSSE(t *testing.T) {
	srv := newStreamTestServer(t)
	resp, err := http.Post(srv.URL+"/a2a", "application/json",
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"message/stream","params":{"agentType":"Poet","agentId":"p1","message":{"role":"user","parts":[{"type":"text","text":"hi"}]}}}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	var events []wire.TaskStatusUpdateEvent
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var ev wire.TaskStatusUpdateEvent
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev); err != nil {
			t.Fatalf("bad SSE data %q: %v", line, err)
		}
		events = append(events, ev)
	}
	if len(events) < 2 {
		t.Fatalf("want >=2 SSE events, got %d: %+v", len(events), events)
	}
	if events[0].Status.State != "working" || events[0].Final {
		t.Fatalf("first event = %+v (want working, not final)", events[0])
	}
	last := events[len(events)-1]
	if !last.Final || last.Status.State != "completed" {
		t.Fatalf("last event = %+v (want final completed)", last)
	}
	if last.Status.Message == nil || last.Status.Message.Parts[0].Text != "hello from fake" {
		t.Fatalf("final answer = %+v", last.Status)
	}
}

func TestTasksCancel(t *testing.T) {
	srv := newStreamTestServer(t)

	// Start an async run that blocks in the fake until canceled.
	var out struct {
		Result wire.Task `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"agentType":"Poet","agentId":"p1","message":{"role":"user","parts":[{"type":"text","text":"BLOCK"}]}}}`, &out)
	taskID := out.Result.ID
	waitForState(t, srv.URL, taskID, "working")

	// Cancel it.
	var c struct {
		Result wire.Task   `json:"result"`
		Error  *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":2,"method":"tasks/cancel","params":{"id":"`+taskID+`"}}`, &c)
	if c.Error != nil {
		t.Fatalf("cancel error: %+v", c.Error)
	}
	if c.Result.Status.State != "canceled" {
		t.Fatalf("cancel response state = %q", c.Result.Status.State)
	}
	// The worker confirms cancellation; the registry reflects it.
	waitForState(t, srv.URL, taskID, "canceled")

	// Re-cancelling a terminal task → TaskNotCancelable.
	var c2 struct {
		Error *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":3,"method":"tasks/cancel","params":{"id":"`+taskID+`"}}`, &c2)
	if c2.Error == nil || c2.Error.Code != wire.TaskNotCancelable {
		t.Fatalf("want TaskNotCancelable, got %+v", c2.Error)
	}

	// Cancelling an unknown task → TaskNotFound.
	var c3 struct {
		Error *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":4,"method":"tasks/cancel","params":{"id":"ghost"}}`, &c3)
	if c3.Error == nil || c3.Error.Code != wire.TaskNotFound {
		t.Fatalf("want TaskNotFound, got %+v", c3.Error)
	}
}

func TestToolsList(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Result struct {
			Tools []wire.MCPToolDef `json:"tools"`
		} `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"agentType":"Poet"}}`, &out)
	if len(out.Result.Tools) != 1 {
		t.Fatalf("tools = %d", len(out.Result.Tools))
	}
	tool := out.Result.Tools[0]
	if tool.Name != "enrich_company" || tool.Description != "Enrich a company." {
		t.Fatalf("tool def = %+v", tool)
	}
	props, _ := tool.InputSchema["properties"].(map[string]any)
	if _, ok := props["company_name"]; !ok {
		t.Fatalf("inputSchema missing company_name: %+v", tool.InputSchema)
	}
}

func TestToolsCall(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Result wire.MCPCallToolResult `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"agentType":"Poet","agentId":"p1","name":"enrich_company","arguments":{"company_name":"Northwind"}}}`, &out)
	if out.Result.IsError {
		t.Fatalf("unexpected isError")
	}
	if !strings.Contains(out.Result.Content[0].Text, "Northwind") {
		t.Fatalf("content = %q", out.Result.Content[0].Text)
	}
}

func TestMethodNotFound(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Error *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"bogus","params":{}}`, &out)
	if out.Error == nil || out.Error.Code != wire.MethodNotFound {
		t.Fatalf("expected method-not-found, got %+v", out.Error)
	}
}

func TestAgentCard(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Get(srv.URL + "/.well-known/agent.json")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var card wire.AgentCard
	if err := json.NewDecoder(resp.Body).Decode(&card); err != nil {
		t.Fatal(err)
	}
	if card.Name != "rumi" || card.ProtocolVersion != "0.3" {
		t.Fatalf("card = %+v", card)
	}
}

// ---- Phase 2c: threads / messages-history / resources ----

func TestThreadsList(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Result struct {
			Threads []wire.Thread `json:"threads"`
		} `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"threads/list","params":{"agentType":"Poet"}}`, &out)
	if len(out.Result.Threads) != 1 {
		t.Fatalf("threads = %d", len(out.Result.Threads))
	}
	th := out.Result.Threads[0]
	if th.ThreadID != "Poet/p1" || th.MessageCount != 2 || th.FirstMessagePreview != "hello there" {
		t.Fatalf("thread = %+v", th)
	}
}

func TestThreadsListNeedsAgentType(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Error *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"threads/list","params":{}}`, &out)
	if out.Error == nil || out.Error.Code != wire.InvalidParams {
		t.Fatalf("want InvalidParams, got %+v", out.Error)
	}
}

func TestThreadsCreateEcho(t *testing.T) {
	srv := newTestServer(t)
	// explicit agentId echoes a usable contextId, zeroed counts, stamped times.
	var out struct {
		Result wire.Thread `json:"result"`
		Error  *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"threads/create","params":{"agentType":"Poet","agentId":"conv9"}}`, &out)
	if out.Error != nil {
		t.Fatalf("error: %+v", out.Error)
	}
	if out.Result.ThreadID != "Poet/conv9" || out.Result.MessageCount != 0 {
		t.Fatalf("thread = %+v", out.Result)
	}
	if out.Result.CreatedAt == "" || out.Result.UpdatedAt == "" {
		t.Fatalf("timestamps missing: %+v", out.Result)
	}
	// omitted agentId is minted into a fresh contextId.
	var out2 struct {
		Result wire.Thread `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":2,"method":"threads/create","params":{"agentType":"Poet"}}`, &out2)
	if !strings.HasPrefix(out2.Result.ThreadID, "Poet/") || out2.Result.ThreadID == "Poet/" {
		t.Fatalf("minted threadId = %q", out2.Result.ThreadID)
	}
}

func TestThreadsCreateRejectsBadAgentID(t *testing.T) {
	srv := newTestServer(t)
	// JSON literals: "a\\b" decodes to a single backslash; three spaces is blank.
	bads := []string{
		`{"jsonrpc":"2.0","id":1,"method":"threads/create","params":{"agentType":"Poet","agentId":"a\\b"}}`,
		`{"jsonrpc":"2.0","id":1,"method":"threads/create","params":{"agentType":"Poet","agentId":"a/b"}}`,
		`{"jsonrpc":"2.0","id":1,"method":"threads/create","params":{"agentType":"Poet","agentId":"."}}`,
		`{"jsonrpc":"2.0","id":1,"method":"threads/create","params":{"agentType":"Poet","agentId":".."}}`,
		`{"jsonrpc":"2.0","id":1,"method":"threads/create","params":{"agentType":"Poet","agentId":"   "}}`,
	}
	for _, body := range bads {
		var out struct {
			Error *wire.Error `json:"error"`
		}
		postA2A(t, srv.URL, body, &out)
		if out.Error == nil || out.Error.Code != wire.InvalidParams {
			t.Fatalf("body %s: want InvalidParams, got %+v", body, out.Error)
		}
	}
}

func TestMessagesHistory(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Result struct {
			Messages []wire.Message `json:"messages"`
			HasMore  bool           `json:"hasMore"`
		} `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"messages/history","params":{"contextId":"Poet/p1"}}`, &out)
	if len(out.Result.Messages) != 2 {
		t.Fatalf("messages = %d", len(out.Result.Messages))
	}
	if out.Result.Messages[0].Role != "user" || out.Result.Messages[0].Parts[0].Text != "hello there" {
		t.Fatalf("msg[0] = %+v", out.Result.Messages[0])
	}
	if out.Result.HasMore {
		t.Fatalf("hasMore should be false")
	}
}

func TestResourcesList(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Result struct {
			Resources []wire.MCPResource `json:"resources"`
		} `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"resources/list","params":{"agentType":"Poet","agentId":"p1"}}`, &out)
	if len(out.Result.Resources) != 1 || out.Result.Resources[0].URI != "rumi://box/demo.key" {
		t.Fatalf("resources = %+v", out.Result.Resources)
	}
	if out.Result.Resources[0].MimeType != "application/json" {
		t.Fatalf("mimeType = %q", out.Result.Resources[0].MimeType)
	}
}

func TestResourcesRead(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Result struct {
			Contents []wire.MCPResourceContent `json:"contents"`
		} `json:"result"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"agentType":"Poet","agentId":"p1","uri":"rumi://box/demo.key"}}`, &out)
	if len(out.Result.Contents) != 1 {
		t.Fatalf("contents = %d", len(out.Result.Contents))
	}
	c := out.Result.Contents[0]
	if c.URI != "rumi://box/demo.key" || c.MimeType != "application/json" || !strings.Contains(c.Text, "world") {
		t.Fatalf("content = %+v", c)
	}
}

func TestResourcesReadNeedsURI(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Error *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"agentType":"Poet","agentId":"p1"}}`, &out)
	if out.Error == nil || out.Error.Code != wire.InvalidParams {
		t.Fatalf("want InvalidParams, got %+v", out.Error)
	}
}

// TestResourcesReadSurfacesWorkerError confirms a worker -32602 (unknown
// resource) passes through with its original code, not flattened to -32603.
func TestResourcesReadSurfacesWorkerError(t *testing.T) {
	srv := newTestServer(t)
	var out struct {
		Error *wire.Error `json:"error"`
	}
	postA2A(t, srv.URL, `{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"agentType":"Poet","agentId":"p1","uri":"rumi://box/nope"}}`, &out)
	if out.Error == nil || out.Error.Code != wire.InvalidParams {
		t.Fatalf("want worker InvalidParams passthrough, got %+v", out.Error)
	}
}
