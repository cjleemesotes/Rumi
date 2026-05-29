package wire

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestTaskRoundTripAndKeys(t *testing.T) {
	task := Task{
		ID:        "abc",
		ContextID: "Poet/p1",
		Status: TaskStatus{
			State:   "completed",
			Message: &Message{Role: "agent", Parts: []Part{{Type: "text", Text: "hi"}}},
		},
	}
	b, err := json.Marshal(task)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{`"contextId":"Poet/p1"`, `"state":"completed"`, `"type":"text"`, `"text":"hi"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
	var got Task
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.Status.Message.Parts[0].Text != "hi" {
		t.Fatalf("roundtrip lost text: %+v", got)
	}
}

func TestMCPCallToolResultKeys(t *testing.T) {
	r := MCPCallToolResult{Content: []MCPContentBlock{{Type: "text", Text: "ok"}}, IsError: false}
	b, _ := json.Marshal(r)
	s := string(b)
	if !strings.Contains(s, `"isError":false`) || !strings.Contains(s, `"content":[`) {
		t.Fatalf("MCP result keys: %s", s)
	}
}

func TestStatusUpdateEventKeys(t *testing.T) {
	ev := TaskStatusUpdateEvent{
		Type:      "status-update",
		TaskID:    "abc",
		ContextID: "Poet/p1",
		Status:    TaskStatus{State: "working"},
		Final:     false,
	}
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	// camelCase keys per the A2A wire contract; `final` is always present
	// (not omitempty) so clients can key on false vs true.
	for _, want := range []string{`"type":"status-update"`, `"taskId":"abc"`, `"contextId":"Poet/p1"`, `"state":"working"`, `"final":false`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
}

func TestArtifactUpdateEventKeys(t *testing.T) {
	ev := TaskArtifactUpdateEvent{
		Type:     "artifact-update",
		TaskID:   "abc",
		Artifact: A2AArtifact{ArtifactID: "a1", Name: "result", Parts: []Part{{Type: "text", Text: "hi"}}},
	}
	b, _ := json.Marshal(ev)
	s := string(b)
	for _, want := range []string{`"type":"artifact-update"`, `"artifactId":"a1"`, `"name":"result"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
}

func TestThreadKeys(t *testing.T) {
	th := Thread{
		ThreadID: "Poet/p1", Title: "p1",
		CreatedAt: "2026-05-29T00:00:00Z", UpdatedAt: "2026-05-29T00:00:01Z",
		MessageCount: 2, FirstMessagePreview: "hi",
	}
	b, _ := json.Marshal(th)
	s := string(b)
	// camelCase per the A2A wire contract; messageCount/firstMessagePreview are
	// NOT omitempty so a zero/empty thread still carries them.
	for _, want := range []string{
		`"threadId":"Poet/p1"`, `"messageCount":2`,
		`"createdAt":"2026-05-29T00:00:00Z"`, `"updatedAt":"2026-05-29T00:00:01Z"`,
		`"firstMessagePreview":"hi"`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
}

func TestResourceKeys(t *testing.T) {
	r := MCPResource{URI: "rumi://box/k", Name: "k", Description: "d", MimeType: "application/json"}
	b, _ := json.Marshal(r)
	s := string(b)
	for _, want := range []string{`"uri":"rumi://box/k"`, `"name":"k"`, `"mimeType":"application/json"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
	rc := MCPResourceContent{URI: "rumi://box/k", Text: "{}", MimeType: "application/json"}
	b2 := string(mustMarshal(t, rc))
	if !strings.Contains(b2, `"mimeType":"application/json"`) || !strings.Contains(b2, `"text":"{}"`) {
		t.Fatalf("resource content keys: %s", b2)
	}
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
