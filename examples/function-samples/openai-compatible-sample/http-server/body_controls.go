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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"
)

const (
	headerBodyDigest          = "X-Load-Tester-Body-Digest"
	headerReadDelayMs         = "X-Load-Tester-Read-Delay-Ms"
	headerReadPauseAfterBytes = "X-Load-Tester-Read-Pause-After-Bytes"
	headerReadPauseMs         = "X-Load-Tester-Read-Pause-Ms"
	headerEarlyStatusCode     = "X-Load-Tester-Early-Status-Code"

	maxRequestBytesEnv  = "LOAD_TESTER_MAX_REQUEST_BYTES"
	readTimeoutEnv      = "LOAD_TESTER_READ_TIMEOUT_MS"
	hardMaxRequestBytes = 1 << 30
	hardMaxReadTimeout  = time.Hour
	defaultReadTimeout  = 30 * time.Second
	pacedReadBytes      = 16 << 10
	receiptTimeLayout   = "2006-01-02T15:04:05.000Z"
)

// bodyConfig holds the server-wide request body settings.
type bodyConfig struct {
	// maxRequestBytes replaces every route's body limit when it is not zero.
	maxRequestBytes int64
	readTimeout     time.Duration
	// log receives one line per POST model request when LOG_REQUESTS=true;
	// nil turns the line off.
	log *slog.Logger
}

func loadBodyConfig(getenv func(string) string) (bodyConfig, error) {
	config := bodyConfig{readTimeout: defaultReadTimeout}
	if getenv(logRequestsEnv) == "true" {
		config.log = slog.New(slog.NewJSONHandler(os.Stderr, nil))
	}
	if value := getenv(maxRequestBytesEnv); value != "" {
		limit, err := strconv.ParseInt(value, 10, 64)
		if err != nil || limit < 1 || limit > hardMaxRequestBytes {
			return bodyConfig{}, fmt.Errorf("%s must be an integer from 1 through %d bytes", maxRequestBytesEnv, hardMaxRequestBytes)
		}
		config.maxRequestBytes = limit
	}
	if value := getenv(readTimeoutEnv); value != "" {
		milliseconds, err := strconv.ParseInt(value, 10, 64)
		if err != nil || milliseconds < 1 || milliseconds > hardMaxReadTimeout.Milliseconds() {
			return bodyConfig{}, fmt.Errorf("%s must be an integer from 1 through %d milliseconds", readTimeoutEnv, hardMaxReadTimeout.Milliseconds())
		}
		config.readTimeout = time.Duration(milliseconds) * time.Millisecond
	}
	return config, nil
}

// limit returns the body limit for a route whose own limit is routeLimit.
func (config bodyConfig) limit(routeLimit int64) int64 {
	if config.maxRequestBytes > 0 {
		return config.maxRequestBytes
	}
	return routeLimit
}

// bodyRoute is a POST route the body controls apply to.
type bodyRoute struct {
	limit      int64
	writeError routeErrorWriter
}

var bodyRoutes = map[string]bodyRoute{
	"/v1/chat/completions": {limit: maxRequestBytes, writeError: openAIRouteError},
	"/v1/completions":      {limit: maxRequestBytes, writeError: openAIRouteError},
	"/v1/embeddings":       {limit: maxRequestBytes, writeError: openAIRouteError},
	"/v1/responses":        {limit: maxRequestBytes, writeError: openAIRouteError},
	"/v1/messages":         {limit: maxRequestBytes, writeError: messagesRouteError},
	decisionRoute:          {limit: decisionMaxRequestBytes, writeError: decisionRouteError(false)},
	systemOneRoute:         {limit: decisionMaxRequestBytes, writeError: decisionRouteError(true)},
}

type bodyLimitKey struct{}

// requestBodyLimit returns the body limit the body controls set for this
// request, or routeLimit when they did not run.
func requestBodyLimit(r *http.Request, routeLimit int64) int64 {
	if limit, ok := r.Context().Value(bodyLimitKey{}).(int64); ok {
		return limit
	}
	return routeLimit
}

// bodyControls are header-only controls that act on the raw request body
// before a route parses it, so the body itself is never changed.
type bodyControls struct {
	digest      bool
	readDelay   time.Duration
	pauseAfter  int64
	pause       time.Duration
	earlyStatus int
}

// bodyReceipt describes the request body as the server read it.
type bodyReceipt struct {
	Object        string `json:"object"`
	Bytes         int64  `json:"bytes"`
	SHA256        string `json:"sha256,omitempty"`
	ContentLength int64  `json:"content_length"`
	Accept        string `json:"accept,omitempty"`
	FirstByteAt   string `json:"first_byte_at,omitempty"`
	LastByteAt    string `json:"last_byte_at,omitempty"`
	EOFAt         string `json:"eof_at,omitempty"`
	Complete      bool   `json:"complete"`
	ReadError     string `json:"read_error,omitempty"`
}

// routeErrorWriter writes an error in the route's own response shape.
type routeErrorWriter func(w http.ResponseWriter, status int, message, code string)

// withBodyControls applies the body controls to the POST model routes and
// logs one line per request describing the body the server read. Other
// requests pass through unchanged.
func withBodyControls(config bodyConfig, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		route, ok := bodyRoutes[r.URL.Path]
		if !ok || r.Method != http.MethodPost {
			next.ServeHTTP(w, r)
			return
		}
		limit := config.limit(route.limit)
		controls, err := parseBodyControls(r)
		body := newObservedBody(http.MaxBytesReader(w, newPacedReader(r.Context(), r.Body, controls), limit), controls.digest)
		if config.log != nil {
			defer logRequestBody(config.log, r, body, controls.earlyStatus)
		}
		if err != nil {
			route.writeError(w, http.StatusBadRequest, err.Error(), "")
			return
		}
		if controls.earlyStatus != 0 {
			// Without full duplex, an HTTP/1 server reads up to 256 KiB of the
			// unread body before it sends the response, so a stalled upload
			// would hold back the early answer.
			_ = http.NewResponseController(w).EnableFullDuplex()
			route.writeError(w, controls.earlyStatus, "injected early response before reading the request body", "injected_early_response")
			return
		}
		r = r.WithContext(context.WithValue(r.Context(), bodyLimitKey{}, limit))
		r.Body = body
		if controls.digest {
			status := http.StatusOK
			if _, err := io.Copy(io.Discard, body); err != nil {
				status = http.StatusBadRequest
				var tooLarge *http.MaxBytesError
				if errors.As(err, &tooLarge) {
					status = http.StatusRequestEntityTooLarge
				}
			}
			writeJSON(w, status, body.receipt(r))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func parseBodyControls(r *http.Request) (bodyControls, error) {
	var controls bodyControls
	var err error
	if controls.digest, _, err = boolControl(r, headerBodyDigest, "", nil); err != nil {
		return bodyControls{}, err
	}
	if controls.readDelay, err = durationControl(r, headerReadDelayMs, "", nil, 0); err != nil {
		return bodyControls{}, err
	}
	pauseAfter, hasPauseAfter, _, err := integerControl(r, headerReadPauseAfterBytes, "", nil, 0, 0, hardMaxRequestBytes)
	if err != nil {
		return bodyControls{}, err
	}
	controls.pauseAfter = int64(pauseAfter)
	if controls.pause, err = durationControl(r, headerReadPauseMs, "", nil, 0); err != nil {
		return bodyControls{}, err
	}
	if _, hasPause, _ := oneHeader(r, headerReadPauseMs); hasPauseAfter != hasPause {
		return bodyControls{}, fmt.Errorf("%s and %s must be set together", headerReadPauseAfterBytes, headerReadPauseMs)
	}
	if controls.earlyStatus, _, _, err = integerControl(r, headerEarlyStatusCode, "", nil, 0, http.StatusBadRequest, 599); err != nil {
		return bodyControls{}, err
	}
	if controls.earlyStatus != 0 && controls.digest {
		return bodyControls{}, fmt.Errorf("%s and %s cannot be combined", headerEarlyStatusCode, headerBodyDigest)
	}
	return controls, nil
}

// pacedReader slows reads from the client to model a backend that applies
// backpressure: an optional delay before every read of at most
// pacedReadBytes, and one pause once pauseAfter bytes have been read. A zero
// pause means no pause.
type pacedReader struct {
	ctx        context.Context
	body       io.ReadCloser
	delay      time.Duration
	pauseAfter int64
	pause      time.Duration
	read       int64
}

func newPacedReader(ctx context.Context, body io.ReadCloser, controls bodyControls) io.ReadCloser {
	if controls.readDelay == 0 && controls.pause == 0 {
		return body
	}
	return &pacedReader{ctx: ctx, body: body, delay: controls.readDelay, pauseAfter: controls.pauseAfter, pause: controls.pause}
}

func (reader *pacedReader) Read(p []byte) (int, error) {
	if reader.pause > 0 && reader.read >= reader.pauseAfter {
		pause := reader.pause
		reader.pause = 0
		if !waitFor(reader.ctx, pause, 0) {
			return 0, reader.ctx.Err()
		}
	}
	if reader.delay > 0 {
		if len(p) > pacedReadBytes {
			p = p[:pacedReadBytes]
		}
		if !waitFor(reader.ctx, reader.delay, 0) {
			return 0, reader.ctx.Err()
		}
	}
	if reader.pause > 0 && int64(len(p)) > reader.pauseAfter-reader.read {
		p = p[:reader.pauseAfter-reader.read]
	}
	n, err := reader.body.Read(p)
	reader.read += int64(n)
	return n, err
}

func (reader *pacedReader) Close() error {
	return reader.body.Close()
}

// observedBody records how many bytes a handler read, when, and how the read
// ended. With the digest control it also hashes the raw bytes as they arrive.
type observedBody struct {
	body      io.ReadCloser
	hash      hash.Hash
	bytes     int64
	firstByte time.Time
	lastByte  time.Time
	eof       time.Time
	readErr   error
}

func newObservedBody(body io.ReadCloser, digest bool) *observedBody {
	observed := &observedBody{body: body}
	if digest {
		observed.hash = sha256.New()
	}
	return observed
}

func (body *observedBody) Read(p []byte) (int, error) {
	n, err := body.body.Read(p)
	now := time.Now()
	if n > 0 {
		if body.firstByte.IsZero() {
			body.firstByte = now
		}
		body.lastByte = now
		body.bytes += int64(n)
		if body.hash != nil {
			body.hash.Write(p[:n])
		}
	}
	if err == io.EOF && body.eof.IsZero() {
		body.eof = now
	} else if err != nil && err != io.EOF && body.readErr == nil {
		body.readErr = err
	}
	return n, err
}

func (body *observedBody) Close() error {
	return body.body.Close()
}

func (body *observedBody) receipt(r *http.Request) bodyReceipt {
	receipt := bodyReceipt{
		Object:        "load_tester.body_digest",
		Bytes:         body.bytes,
		ContentLength: r.ContentLength,
		Accept:        r.Header.Get("Accept"),
		FirstByteAt:   receiptTime(body.firstByte),
		LastByteAt:    receiptTime(body.lastByte),
		EOFAt:         receiptTime(body.eof),
		Complete:      !body.eof.IsZero() && body.readErr == nil,
	}
	if body.hash != nil {
		receipt.SHA256 = hex.EncodeToString(body.hash.Sum(nil))
	}
	if body.readErr != nil {
		receipt.ReadError = body.readErr.Error()
	}
	return receipt
}

func logRequestBody(logger *slog.Logger, r *http.Request, body *observedBody, earlyStatus int) {
	receipt := body.receipt(r)
	attrs := []any{"method", r.Method, "path", r.URL.Path}
	for _, field := range []struct{ key, value string }{
		{"request_id", r.Header.Get("NVCF-REQID")},
		{"function_id", r.Header.Get("NVCF-FUNCTION-ID")},
	} {
		if field.value != "" {
			attrs = append(attrs, field.key, field.value)
		}
	}
	attrs = append(attrs, "content_length", receipt.ContentLength, "bytes", receipt.Bytes, "complete", receipt.Complete)
	for _, field := range []struct{ key, value string }{
		{"sha256", receipt.SHA256},
		{"first_byte_at", receipt.FirstByteAt},
		{"last_byte_at", receipt.LastByteAt},
		{"eof_at", receipt.EOFAt},
		{"read_error", receipt.ReadError},
	} {
		if field.value != "" {
			attrs = append(attrs, field.key, field.value)
		}
	}
	if earlyStatus != 0 {
		attrs = append(attrs, "early_status", earlyStatus)
	}
	logger.Info("request body", attrs...)
}

func receiptTime(value time.Time) string {
	if value.IsZero() {
		return ""
	}
	return value.UTC().Format(receiptTimeLayout)
}

func openAIRouteError(w http.ResponseWriter, status int, message, code string) {
	writeAPIErrorWithCode(w, status, message, "", code)
}

func messagesRouteError(w http.ResponseWriter, status int, message, _ string) {
	requestID := fmt.Sprintf("req_%d", responseSequence.Add(1))
	w.Header().Set("request-id", requestID)
	writeMessagesError(w, status, message, requestID)
}

func decisionRouteError(systemOne bool) routeErrorWriter {
	return func(w http.ResponseWriter, status int, message, _ string) {
		writeDecisionError(w, systemOne, status, message)
	}
}
