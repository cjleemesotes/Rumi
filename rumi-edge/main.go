// Command rumi-edge is the A2A + MCP daemon: it serves JSON-RPC over HTTP and
// forwards agent work to a Python worker over a loopback socket.
package main

import (
	"flag"
	"log"
	"net/http"

	"rumi-edge/internal/server"
	"rumi-edge/internal/worker"
)

func main() {
	addr := flag.String("addr", ":8080", "HTTP listen address")
	workerAddr := flag.String("worker-addr", "127.0.0.1:8765", "Python worker address (host:port)")
	flag.Parse()

	s := server.New(worker.New(*workerAddr))
	log.Printf("rumi-edge listening on %s — worker at %s", *addr, *workerAddr)
	if err := http.ListenAndServe(*addr, s.Routes()); err != nil {
		log.Fatal(err)
	}
}
