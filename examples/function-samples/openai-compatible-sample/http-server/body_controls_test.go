// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// modelRoutes are the POST routes the body controls apply to.
var modelRoutes = []string{
	"/v1/chat/completions",
	"/v1/completions",
	"/v1/embeddings",
	"/v1/responses",
	"/v1/messages",
	decisionRoute,
	systemOneRoute,
}

type testReceipt struct {
	Object        string `json:"object"`
	Bytes         int64  `json:"bytes"`
	SHA256        string `json:"sha256"`
	ContentLength int64  `json:"content_length"`
	Accept        string `json:"accept"`
	FirstByteAt   string `json:"first_byte_at"`
	LastByteAt    string `json:"last_byte_at"`
	EOFAt         string `json:"eof_at"`
	Complete      bool   `json:"complete"`
	ReadError     string `json:"read_error"`
}

func TestBodyDigestHashesRawBytes(t *testing.T) {
	// Duplicate keys and odd whitespace disappear in any JSON round trip.
	body := "{\"model\":\"a\",  \"model\":\"b\",\n\"stream\":false,\t\"stream\":true,\"messages\":[]}"
	router, _ := newTestRouter(nil)
	request := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	request.Header.Set(headerBodyDigest, "true")
	request.Header.Set("Accept", "text/event-stream")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", recorder.Code, recorder.Body.String())
	}
	receipt := decodeReceipt(t, recorder.Body.Bytes())
	want := testReceipt{
		Object:        "load_tester.body_digest",
		Bytes:         int64(len(body)),
		SHA256:        sha256Hex(body),
		ContentLength: int64(len(body)),
		Accept:        "text/event-stream",
		Complete:      true,
	}
	receipt.FirstByteAt, receipt.LastByteAt, receipt.EOFAt = "", "", ""
	if receipt != want {
		t.Fatalf("receipt = %+v, want %+v", receipt, want)
	}
}

func TestBodyDigestCoversEveryModelRouteWithoutParsing(t *testing.T) {
	body := `{"model":"test-model","stream":false,"padding":"` + strings.Repeat("p", 70000) + `",` + "\x00 not json"
	router, _ := newTestRouter(nil)
	for _, path := range modelRoutes {
		t.Run(path, func(t *testing.T) {
			request := httptest.NewRequestWithContext(context.Background(), http.MethodPost, path, strings.NewReader(body))
			request.Header.Set(headerBodyDigest, "true")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, request)
			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200: %s", recorder.Code, recorder.Body.String())
			}
			receipt := decodeReceipt(t, recorder.Body.Bytes())
			if receipt.SHA256 != sha256Hex(body) || receipt.Bytes != int64(len(body)) || !receipt.Complete {
				t.Fatalf("receipt = %+v, want complete %d bytes with sha256 %s", receipt, len(body), sha256Hex(body))
			}
		})
	}
}

func TestBodyDigestTimestampsSpanClientPause(t *testing.T) {
	router, _ := newTestRouter(nil)
	server := httptest.NewServer(router)
	defer server.Close()

	reader, writer := io.Pipe()
	go func() {
		_, _ = writer.Write([]byte(`{"model":"test-model",`))
		// The margin over the 250ms check absorbs a late first server read.
		time.Sleep(600 * time.Millisecond)
		_, _ = writer.Write([]byte(`"messages":[]}`))
		_ = writer.Close()
	}()
	receipt := postReceipt(t, server.URL+"/v1/chat/completions", reader, map[string]string{headerBodyDigest: "true"})

	if receipt.ContentLength != -1 {
		t.Fatalf("content_length = %d, want -1 for a chunked upload", receipt.ContentLength)
	}
	first := parseReceiptTime(t, receipt.FirstByteAt)
	eof := parseReceiptTime(t, receipt.EOFAt)
	if gap := eof.Sub(first); gap < 250*time.Millisecond {
		t.Fatalf("eof_at - first_byte_at = %s, want most of the 600ms client pause", gap)
	}
}

func TestBodyDigestKeepsBodyCap(t *testing.T) {
	router, _ := newTestRouter(func(config *serverConfig) {
		config.body.maxRequestBytes = 1024
	})
	for _, path := range []string{"/v1/chat/completions", decisionRoute} {
		t.Run(path, func(t *testing.T) {
			request := httptest.NewRequestWithContext(context.Background(), http.MethodPost, path, strings.NewReader(strings.Repeat("a", 2048)))
			request.Header.Set(headerBodyDigest, "true")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, request)
			if recorder.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("status = %d, want 413: %s", recorder.Code, recorder.Body.String())
			}
			receipt := decodeReceipt(t, recorder.Body.Bytes())
			if receipt.Complete || receipt.Bytes != 1024 || !strings.Contains(receipt.ReadError, "too large") {
				t.Fatalf("receipt = %+v, want incomplete 1024 bytes with a too large error", receipt)
			}
		})
	}
}

func TestReadPauseHoldsReadsAfterByteCount(t *testing.T) {
	router, _ := newTestRouter(nil)
	server := httptest.NewServer(router)
	defer server.Close()
	headers := map[string]string{
		headerBodyDigest:          "true",
		headerReadPauseAfterBytes: "1024",
		headerReadPauseMs:         "300",
	}

	t.Run("body longer than the pause point", func(t *testing.T) {
		body := strings.Repeat("a", 4096)
		receipt := postReceipt(t, server.URL+"/v1/chat/completions", strings.NewReader(body), headers)
		if receipt.SHA256 != sha256Hex(body) || !receipt.Complete {
			t.Fatalf("receipt = %+v, want the complete body", receipt)
		}
		if gap := parseReceiptTime(t, receipt.EOFAt).Sub(parseReceiptTime(t, receipt.FirstByteAt)); gap < 250*time.Millisecond {
			t.Fatalf("eof_at - first_byte_at = %s, want at least the 300ms pause", gap)
		}
	})
	t.Run("body shorter than the pause point", func(t *testing.T) {
		receipt := postReceipt(t, server.URL+"/v1/chat/completions", strings.NewReader(strings.Repeat("a", 512)), headers)
		if gap := parseReceiptTime(t, receipt.EOFAt).Sub(parseReceiptTime(t, receipt.FirstByteAt)); gap >= 250*time.Millisecond {
			t.Fatalf("eof_at - first_byte_at = %s, want no pause before byte 1024", gap)
		}
	})
}

func TestReadPauseStartsExactlyAtByteCount(t *testing.T) {
	router, logs := newTestRouter(nil)
	server := httptest.NewServer(router)
	defer server.Close()

	// The client sends only the first sent bytes of a 4096-byte body, waits
	// longer than the 300ms pause, then closes. A byte read after the pause
	// shows up as a gap between the first and last byte times.
	for name, test := range map[string]struct {
		sent    int
		wantGap bool
	}{
		"no byte past the pause point":  {sent: 1024, wantGap: false},
		"one byte past the pause point": {sent: 1025, wantGap: true},
	} {
		t.Run(name, func(t *testing.T) {
			requestID := fmt.Sprintf("pause-%d", test.sent)
			connection, err := net.Dial("tcp", strings.TrimPrefix(server.URL, "http://"))
			if err != nil {
				t.Fatal(err)
			}
			_, _ = fmt.Fprintf(connection, "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nNVCF-REQID: %s\r\n%s: 1024\r\n%s: 300\r\nContent-Length: 4096\r\n\r\n%s",
				requestID, headerReadPauseAfterBytes, headerReadPauseMs, strings.Repeat("a", test.sent))
			time.Sleep(600 * time.Millisecond)
			_ = connection.Close()

			line := waitLogLine(t, logs, requestID)
			if line["bytes"] != float64(test.sent) {
				t.Fatalf("log bytes = %v, want %d", line["bytes"], test.sent)
			}
			gap := parseReceiptTime(t, fmt.Sprint(line["last_byte_at"])).Sub(parseReceiptTime(t, fmt.Sprint(line["first_byte_at"])))
			if gotGap := gap >= 250*time.Millisecond; gotGap != test.wantGap {
				t.Fatalf("last_byte_at - first_byte_at = %s, want a pause gap %t", gap, test.wantGap)
			}
		})
	}
}

func TestReadDelayPacesEveryRead(t *testing.T) {
	router, _ := newTestRouter(nil)
	server := httptest.NewServer(router)
	defer server.Close()

	// Reads are at most pacedReadBytes, so 4 full reads plus 1 byte need 5
	// data reads and at least 4 delays between the first and last byte.
	body := strings.Repeat("a", 4*pacedReadBytes+1)
	receipt := postReceipt(t, server.URL+"/v1/chat/completions", strings.NewReader(body), map[string]string{
		headerBodyDigest:  "true",
		headerReadDelayMs: "100",
	})
	if receipt.SHA256 != sha256Hex(body) {
		t.Fatalf("receipt = %+v, want sha256 of the body", receipt)
	}
	if gap := parseReceiptTime(t, receipt.LastByteAt).Sub(parseReceiptTime(t, receipt.FirstByteAt)); gap < 350*time.Millisecond {
		t.Fatalf("last_byte_at - first_byte_at = %s, want at least 4 read delays of 100ms", gap)
	}
}

func TestRejectsInvalidBodyControlHeaders(t *testing.T) {
	for name, headers := range map[string]map[string]string{
		"digest not boolean":       {headerBodyDigest: "maybe"},
		"negative read delay":      {headerReadDelayMs: "-1"},
		"read delay too long":      {headerReadDelayMs: fmt.Sprint(maxControlMilliseconds + 1)},
		"pause bytes without time": {headerReadPauseAfterBytes: "10"},
		"pause time without bytes": {headerReadPauseMs: "10"},
		"pause bytes not a number": {headerReadPauseAfterBytes: "ten", headerReadPauseMs: "10"},
		"early status not error":   {headerEarlyStatusCode: "200"},
		"early status not number":  {headerEarlyStatusCode: "abc"},
		"early status with digest": {headerEarlyStatusCode: "413", headerBodyDigest: "true"},
	} {
		t.Run(name, func(t *testing.T) {
			router, _ := newTestRouter(nil)
			request := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/v1/chat/completions", unreadBody{t: t})
			for header, value := range headers {
				request.Header.Set(header, value)
			}
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, request)
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", recorder.Code, recorder.Body.String())
			}
			if message := nestedString(t, responseMap(t, recorder), "error", "message"); message == "" {
				t.Fatalf("error message is empty: %s", recorder.Body.String())
			}
		})
	}
}

func TestEarlyStatusAnswersBeforeReadingBody(t *testing.T) {
	for path, check := range map[string]func(*testing.T, map[string]any){
		"/v1/chat/completions": func(t *testing.T, body map[string]any) {
			if got := nestedString(t, body, "error", "code"); got != "injected_early_response" {
				t.Fatalf("error.code = %q, want injected_early_response", got)
			}
		},
		"/v1/messages": func(t *testing.T, body map[string]any) {
			if got := nestedString(t, body, "error", "type"); got != "request_too_large" {
				t.Fatalf("error.type = %q, want request_too_large", got)
			}
		},
		decisionRoute: func(t *testing.T, body map[string]any) {
			if got := nestedString(t, body, "error", "code"); got != "413" {
				t.Fatalf("error.code = %q, want 413", got)
			}
		},
		systemOneRoute: func(t *testing.T, body map[string]any) {
			if _, ok := body["detail"].([]any); !ok {
				t.Fatalf("body = %v, want a FastAPI detail list", body)
			}
		},
	} {
		t.Run(path, func(t *testing.T) {
			router, logs := newTestRouter(nil)
			request := httptest.NewRequestWithContext(context.Background(), http.MethodPost, path, unreadBody{t: t})
			request.Header.Set(headerEarlyStatusCode, "413")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, request)
			if recorder.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("status = %d, want 413: %s", recorder.Code, recorder.Body.String())
			}
			check(t, responseMap(t, recorder))
			line := onlyLogLine(t, logs)
			if line["bytes"] != float64(0) || line["complete"] != false || line["early_status"] != float64(413) {
				t.Fatalf("log line = %v, want 0 bytes, incomplete, early_status 413", line)
			}
		})
	}
}

func TestEarlyStatusReachesClientDuringStalledChunkedUpload(t *testing.T) {
	router, _ := newTestRouter(nil)
	server := httptest.NewServer(router)
	defer server.Close()

	connection, err := net.Dial("tcp", strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	// One chunk, then no terminating chunk: the upload stays open.
	_, _ = fmt.Fprintf(connection, "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n%s: 413\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n{\"a\"\r\n", headerEarlyStatusCode)
	_ = connection.SetReadDeadline(time.Now().Add(2 * time.Second))
	status, err := bufio.NewReader(connection).ReadString('\n')
	if err != nil {
		t.Fatalf("no response while the upload is open: %v", err)
	}
	if !strings.Contains(status, " 413 ") {
		t.Fatalf("status line = %q, want 413", status)
	}
}

func TestConfiguredBodyCapAppliesToEveryRoute(t *testing.T) {
	router, _ := newTestRouter(func(config *serverConfig) {
		config.body.maxRequestBytes = 1024
	})
	defaultRouter, _ := newTestRouter(nil)
	padding := `,"pad":"` + strings.Repeat("a", 2048) + `"}`
	for path, prefix := range map[string]string{
		"/v1/chat/completions": `{"model":"test-model","messages":[{"role":"user","content":"hi"}]`,
		"/v1/completions":      `{"model":"test-model","prompt":"hi"`,
		"/v1/embeddings":       `{"model":"test-model","input":"hi"`,
		"/v1/responses":        `{"model":"test-model","input":"hi"`,
		"/v1/messages":         `{"model":"test-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]`,
		decisionRoute:          `{"model":"test-model"`,
		systemOneRoute:         `{"model":"test-model"`,
	} {
		t.Run(path, func(t *testing.T) {
			body := prefix + padding
			control := httptest.NewRecorder()
			defaultRouter.ServeHTTP(control, httptest.NewRequestWithContext(context.Background(), http.MethodPost, path, strings.NewReader(body)))
			if control.Code != http.StatusOK {
				t.Fatalf("default cap status = %d, want 200: %s", control.Code, control.Body.String())
			}
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, httptest.NewRequestWithContext(context.Background(), http.MethodPost, path, strings.NewReader(body)))
			if recorder.Code != http.StatusBadRequest && recorder.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("status = %d, want the route's too large error: %s", recorder.Code, recorder.Body.String())
			}
		})
	}

	t.Run("raised cap accepts a body above the default", func(t *testing.T) {
		router, _ := newTestRouter(func(config *serverConfig) {
			config.body.maxRequestBytes = 12 << 20
		})
		body := `{"model":"test-model","messages":[{"role":"user","content":"` + strings.Repeat("a", 11<<20) + `"}]}`
		request := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200: %s", recorder.Code, recorder.Body.String())
		}
	})
}

func TestLoadServerConfigBodyLimits(t *testing.T) {
	for _, test := range []struct {
		name         string
		env          map[string]string
		wantBytes    int64
		wantDecision int64
		wantTimeout  time.Duration
		wantLog      bool
		wantErr      bool
	}{
		{name: "unset keeps defaults", wantBytes: maxRequestBytes, wantDecision: decisionMaxRequestBytes, wantTimeout: 30 * time.Second},
		{
			name:         "overrides apply to every route",
			env:          map[string]string{maxRequestBytesEnv: "33554432", readTimeoutEnv: "200000"},
			wantBytes:    32 << 20,
			wantDecision: 32 << 20,
			wantTimeout:  200 * time.Second,
		},
		{
			name:         "request log only when enabled",
			env:          map[string]string{logRequestsEnv: "true"},
			wantBytes:    maxRequestBytes,
			wantDecision: decisionMaxRequestBytes,
			wantTimeout:  30 * time.Second,
			wantLog:      true,
		},
		{name: "zero bytes rejected", env: map[string]string{maxRequestBytesEnv: "0"}, wantErr: true},
		{name: "bytes above hard maximum rejected", env: map[string]string{maxRequestBytesEnv: fmt.Sprint(hardMaxRequestBytes + 1)}, wantErr: true},
		{name: "non numeric bytes rejected", env: map[string]string{maxRequestBytesEnv: "lots"}, wantErr: true},
		{name: "zero timeout rejected", env: map[string]string{readTimeoutEnv: "0"}, wantErr: true},
		{name: "timeout above hard maximum rejected", env: map[string]string{readTimeoutEnv: fmt.Sprint(hardMaxReadTimeout.Milliseconds() + 1)}, wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			config, err := loadServerConfig(func(name string) string { return test.env[name] })
			if test.wantErr {
				if err == nil {
					t.Fatal("loadServerConfig() error = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("loadServerConfig() error = %v", err)
			}
			body := config.body
			if got, gotDecision := body.limit(maxRequestBytes), body.limit(decisionMaxRequestBytes); got != test.wantBytes || gotDecision != test.wantDecision {
				t.Fatalf("limits = %d/%d, want %d/%d", got, gotDecision, test.wantBytes, test.wantDecision)
			}
			if body.readTimeout != test.wantTimeout {
				t.Fatalf("read timeout = %s, want %s", body.readTimeout, test.wantTimeout)
			}
			if (body.log != nil) != test.wantLog {
				t.Fatalf("request log enabled = %t, want %t", body.log != nil, test.wantLog)
			}
		})
	}
}

func TestRequestBodyLogLine(t *testing.T) {
	router, logs := newTestRouter(nil)
	server := httptest.NewServer(router)
	defer server.Close()

	t.Run("complete body", func(t *testing.T) {
		body := `{"model":"test-model","messages":[]}`
		request, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("NVCF-REQID", "req-123")
		request.Header.Set("NVCF-FUNCTION-ID", "fn-456")
		response, err := http.DefaultClient.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, response.Body)
		_ = response.Body.Close()

		line := waitLogLine(t, logs, "req-123")
		for key, want := range map[string]any{
			"msg": "request body", "method": "POST", "path": "/v1/chat/completions",
			"request_id": "req-123", "function_id": "fn-456",
			"bytes": float64(len(body)), "content_length": float64(len(body)), "complete": true,
		} {
			if line[key] != want {
				t.Fatalf("log %s = %v, want %v: %v", key, line[key], want, line)
			}
		}
		for _, key := range []string{"first_byte_at", "last_byte_at", "eof_at"} {
			parseReceiptTime(t, fmt.Sprint(line[key]))
		}
		if _, ok := line["read_error"]; ok {
			t.Fatalf("log line has read_error for a complete body: %v", line)
		}
	})

	t.Run("client closes mid body", func(t *testing.T) {
		connection, err := net.Dial("tcp", strings.TrimPrefix(server.URL, "http://"))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = fmt.Fprintf(connection, "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nNVCF-REQID: req-abort\r\nContent-Length: 1000\r\n\r\n%s", strings.Repeat("a", 100))
		time.Sleep(100 * time.Millisecond)
		_ = connection.Close()

		line := waitLogLine(t, logs, "req-abort")
		if line["bytes"] != float64(100) || line["complete"] != false || line["read_error"] == nil {
			t.Fatalf("log line = %v, want 100 bytes, incomplete, with read_error", line)
		}
		if _, ok := line["eof_at"]; ok {
			t.Fatalf("log line has eof_at for an incomplete body: %v", line)
		}
	})

	t.Run("no line for GET", func(t *testing.T) {
		before := len(logLines(t, logs))
		response, err := http.Get(server.URL + "/health")
		if err != nil {
			t.Fatal(err)
		}
		_ = response.Body.Close()
		time.Sleep(50 * time.Millisecond)
		if after := len(logLines(t, logs)); after != before {
			t.Fatalf("log lines = %d after GET, want %d", after, before)
		}
	})
}

// unreadBody fails the test if the handler reads it.
type unreadBody struct {
	t *testing.T
}

func (body unreadBody) Read([]byte) (int, error) {
	body.t.Error("request body was read")
	return 0, io.EOF
}

type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (buffer *syncBuffer) Write(p []byte) (int, error) {
	buffer.mu.Lock()
	defer buffer.mu.Unlock()
	return buffer.buf.Write(p)
}

func (buffer *syncBuffer) String() string {
	buffer.mu.Lock()
	defer buffer.mu.Unlock()
	return buffer.buf.String()
}

func newTestRouter(mutate func(*serverConfig)) (http.Handler, *syncBuffer) {
	logs := &syncBuffer{}
	config := defaultServerConfig()
	config.body.log = slog.New(slog.NewJSONHandler(logs, nil))
	if mutate != nil {
		mutate(&config)
	}
	return newRouterWithConfig(config), logs
}

func postReceipt(t *testing.T, url string, body io.Reader, headers map[string]string) testReceipt {
	t.Helper()
	request, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, body)
	if err != nil {
		t.Fatal(err)
	}
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", response.StatusCode, raw)
	}
	return decodeReceipt(t, raw)
}

func decodeReceipt(t *testing.T, raw []byte) testReceipt {
	t.Helper()
	var receipt testReceipt
	if err := json.Unmarshal(raw, &receipt); err != nil {
		t.Fatalf("receipt JSON unmarshal failed: %v: %s", err, raw)
	}
	return receipt
}

func parseReceiptTime(t *testing.T, value string) time.Time {
	t.Helper()
	parsed, err := time.Parse(receiptTimeLayout, value)
	if err != nil || !strings.HasSuffix(value, "Z") {
		t.Fatalf("timestamp %q is not UTC with milliseconds: %v", value, err)
	}
	return parsed
}

func sha256Hex(body string) string {
	sum := sha256.Sum256([]byte(body))
	return hex.EncodeToString(sum[:])
}

func nestedString(t *testing.T, body map[string]any, object, field string) string {
	t.Helper()
	inner, ok := body[object].(map[string]any)
	if !ok {
		t.Fatalf("body has no %s object: %v", object, body)
	}
	value, _ := inner[field].(string)
	return value
}

func logLines(t *testing.T, logs *syncBuffer) []map[string]any {
	t.Helper()
	var lines []map[string]any
	scanner := bufio.NewScanner(strings.NewReader(logs.String()))
	for scanner.Scan() {
		var line map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &line); err != nil {
			t.Fatalf("log line is not JSON: %v: %s", err, scanner.Text())
		}
		lines = append(lines, line)
	}
	return lines
}

func onlyLogLine(t *testing.T, logs *syncBuffer) map[string]any {
	t.Helper()
	lines := logLines(t, logs)
	if len(lines) != 1 {
		t.Fatalf("log lines = %d, want 1: %s", len(lines), logs.String())
	}
	return lines[0]
}

func waitLogLine(t *testing.T, logs *syncBuffer, requestID string) map[string]any {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for _, line := range logLines(t, logs) {
			if line["request_id"] == requestID {
				return line
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("no log line for request %s: %s", requestID, logs.String())
	return nil
}
