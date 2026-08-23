#!/usr/bin/env python3
"""Small JSON-RPC client for the reussir-lsp lit test."""

import json
import os
import pathlib
import subprocess
import sys


class Client:
    def __init__(self, executable):
        self.process = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.next_id = 1

    def send(self, message):
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        self.process.stdin.write(frame)
        self.process.stdin.flush()

    def notify(self, method, params):
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method, params):
        request_id = self.next_id
        self.next_id += 1
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        while True:
            message = self.read()
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method} failed: {message['error']}")
            return message.get("result")

    def read(self):
        headers = {}
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = self.process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"language server closed its output: {stderr}")
            if line == b"\r\n":
                break
            name, value = line.decode("ascii").split(":", 1)
            headers[name.lower()] = value.strip()
        length = int(headers["content-length"])
        return json.loads(self.process.stdout.read(length))

    def close(self):
        self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            raise RuntimeError("language server did not exit")
        if return_code:
            stderr = self.process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"language server exited with {return_code}: {stderr}")


def decode_tokens(source, legend, result):
    lines = source.splitlines()
    data = result["data"]
    decoded = []
    line = 0
    start = 0
    for index in range(0, len(data), 5):
        delta_line, delta_start, length, token_type, modifiers = data[index : index + 5]
        line += delta_line
        start = start + delta_start if delta_line == 0 else delta_start
        units = lines[line].encode("utf-16-le")
        text = units[start * 2 : (start + length) * 2].decode("utf-16-le")
        decoded.append((text, legend[token_type], line, start, modifiers))
    return decoded


def require(tokens, text, token_type, modifiers=None):
    matches = [
        token
        for token in tokens
        if token[0] == text
        and token[1] == token_type
        and (modifiers is None or token[4] == modifiers)
    ]
    if not matches:
        suffix = "" if modifiers is None else f" with modifiers {modifiers}"
        raise AssertionError(
            f"missing {text!r} as {token_type}{suffix}: {tokens}"
        )
    print(f"TOKEN {text} {token_type}")


def main():
    executable, source_path = sys.argv[1:]
    source_path = pathlib.Path(source_path).resolve()
    source = source_path.read_text(encoding="utf-8")
    uri = source_path.as_uri()
    client = Client(executable)
    try:
        initialized = client.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": source_path.parent.as_uri(),
                "capabilities": {},
            },
        )
        capabilities = initialized["capabilities"]
        sync = capabilities["textDocumentSync"]
        semantic = capabilities["semanticTokensProvider"]
        assert capabilities["positionEncoding"] == "utf-16"
        assert sync == {"openClose": True, "change": 1}
        assert semantic["full"] is True
        assert "range" not in semantic
        print("CAPABILITIES full-sync full-semantic")

        client.notify("initialized", {})
        client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "reussir",
                    "version": 1,
                    "text": source,
                }
            },
        )
        params = {"textDocument": {"uri": uri}}
        result = client.request("textDocument/semanticTokens/full", params)
        legend = semantic["legend"]["tokenTypes"]
        tokens = decode_tokens(source, legend, result)
        # The non-ASCII comment before `main` ensures the server's byte ranges
        # are converted to the UTF-16 columns negotiated above. Bit zero is the
        # standard semantic-token `declaration` modifier.
        require(tokens, "main", "function", modifiers=1)
        require(tokens, "arith.constant", "function")
        require(tokens, "use", "keyword")
        require(tokens, "new", "function")

        changed = "fn changed() -> i64 { 2 }\n"
        client.notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": 2},
                "contentChanges": [{"text": changed}],
            },
        )
        changed_result = client.request("textDocument/semanticTokens/full", params)
        changed_tokens = decode_tokens(changed, legend, changed_result)
        require(changed_tokens, "changed", "function")
        assert not any(token[0] in ("main", "arith.constant") for token in changed_tokens)
        print("CHANGE replaced-whole-document")

        client.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
        assert client.request("textDocument/semanticTokens/full", params) is None
        print("CLOSE returned-null")
        client.request("shutdown", None)
        client.notify("exit", None)
        client.close()
    except Exception:
        client.process.kill()
        stderr = client.process.stderr.read().decode("utf-8", errors="replace")
        if stderr:
            print(stderr, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
