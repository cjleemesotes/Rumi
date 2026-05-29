package server

import "sync"

// taskRecord is the in-memory state of one A2A task. A copy is returned by Get;
// callers never hold the pointer.
type taskRecord struct {
	State     string // submitted|working|completed|failed|canceled
	ContextID string
	Answer    string // final answer text (completed) or error message (failed)
	Metadata  map[string]any
	Updated   string  // ISO timestamp of the last state change
	cancel    func()  // stops the underlying run (worker cancel)
}

func isTerminal(state string) bool {
	return state == "completed" || state == "failed" || state == "canceled"
}

// TaskRegistry is a thread-safe in-memory store of task state. rumi-edge is a
// single-tenant local daemon, so a plain map suffices (unbounded — there is no
// eviction; a long-lived server with many tasks would grow it, which is an
// acceptable trade for v1 simplicity).
type TaskRegistry struct {
	mu sync.Mutex
	m  map[string]*taskRecord
}

func NewTaskRegistry() *TaskRegistry { return &TaskRegistry{m: map[string]*taskRecord{}} }

// Submit registers a new task in the "submitted" state with its cancel func.
func (r *TaskRegistry) Submit(id, ctxID string, cancel func()) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.m[id] = &taskRecord{State: "submitted", ContextID: ctxID, Updated: nowISO(), cancel: cancel}
}

// SetState advances a task to a working (non-terminal) state. Terminal-guarded:
// once a record is completed/failed/canceled its state is frozen, so the first
// terminal transition wins — this is the single invariant that kills the
// cancel-vs-completion race.
func (r *TaskRegistry) SetState(id, state string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if rec := r.m[id]; rec != nil && !isTerminal(rec.State) {
		rec.State = state
		rec.Updated = nowISO()
	}
}

// SetCompleted freezes the task as completed with its answer + metadata.
func (r *TaskRegistry) SetCompleted(id, answer string, meta map[string]any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if rec := r.m[id]; rec != nil && !isTerminal(rec.State) {
		rec.State = "completed"
		rec.Answer = answer
		rec.Metadata = meta
		rec.Updated = nowISO()
	}
}

// SetFailed freezes the task as failed with an error message.
func (r *TaskRegistry) SetFailed(id, msg string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if rec := r.m[id]; rec != nil && !isTerminal(rec.State) {
		rec.State = "failed"
		rec.Answer = msg
		rec.Updated = nowISO()
	}
}

// SetCanceled freezes the task as canceled.
func (r *TaskRegistry) SetCanceled(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if rec := r.m[id]; rec != nil && !isTerminal(rec.State) {
		rec.State = "canceled"
		rec.Updated = nowISO()
	}
}

// Get returns a copy of the task record (and whether it exists).
func (r *TaskRegistry) Get(id string) (taskRecord, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	rec := r.m[id]
	if rec == nil {
		return taskRecord{}, false
	}
	return *rec, true
}

// Cancel fires the task's cancel func if it is still cancelable (exists and is
// non-terminal). Returns (cancelable, found). It does NOT set the state — the
// worker's terminal "canceled" frame drives the authoritative transition, so
// tasks/get reflects reality rather than an optimistic guess.
func (r *TaskRegistry) Cancel(id string) (cancelable, found bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	rec := r.m[id]
	if rec == nil {
		return false, false
	}
	if isTerminal(rec.State) {
		return false, true
	}
	if rec.cancel != nil {
		rec.cancel()
	}
	return true, true
}
