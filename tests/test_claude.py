from grokbot import claude


def delta(index: int, text: str) -> dict:
    return {"type": "stream_event", "event": {
        "type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}}


def block(kind: str, index: int, block_type: str = "text") -> dict:
    event = {"type": kind, "index": index}
    if kind == "content_block_start":
        event["content_block"] = {"type": block_type}
    return {"type": "stream_event", "event": event}


def test_stream_parser_text_tools_and_results():
    parser = claude.StreamParser()
    messages = [
        {"type": "system", "subtype": "init", "session_id": "s1", "model": "claude-haiku-5-5", "apiKeySource": "none"},
        {"type": "stream_event", "event": {"type": "message_start"}},
        block("content_block_start", 0),
        delta(0, "Hi "),
        delta(0, "there."),
        block("content_block_stop", 0),
        block("content_block_start", 1, "tool_use"),
        {"type": "stream_event", "event": {
            "type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{"}}},
        block("content_block_stop", 1),
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Hi there."},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "C:\\proj\\src\\app.py"}},
        ]}},
        {"type": "result", "subtype": "success", "is_error": False, "session_id": "s1", "permission_denials": []},
    ]
    events = [event for message in messages for event in parser.parse(message)]
    assert events == [
        claude.Ready("s1", "claude-haiku-5-5", "none"),
        claude.TextDelta("Hi "),
        claude.TextDelta("there."),
        claude.TextDone("Hi there."),
        claude.ToolCall("Read", "app.py"),
        claude.TurnDone(ok=True, subtype="success", session_id="s1", denials=0),
    ]


def test_permission_request_and_usage():
    parser = claude.StreamParser()
    [ask] = parser.parse({"type": "control_request", "request_id": "r1", "request": {
        "subtype": "can_use_tool", "tool_name": "Bash",
        "input": {"command": "npm test", "description": "Run tests"},
        "permission_suggestions": [{"type": "setMode", "mode": "acceptEdits", "destination": "session"}],
    }})
    assert (ask.request_id, ask.tool, ask.detail) == ("r1", "Bash", "npm test")
    assert ask.suggestions[0]["mode"] == "acceptEdits"
    [usage] = parser.parse({"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed", "rateLimitType": "five_hour", "unifiedWindows": {"five_hour": {"utilization": 0.42}}}})
    assert usage == claude.Usage("allowed", "five_hour", 0.42)


def test_thinking_results_and_modes():
    parser = claude.StreamParser()
    messages = [
        {"type": "system", "subtype": "init", "session_id": "s1", "model": "m", "apiKeySource": "none",
         "permissionMode": "auto"},
        block("content_block_start", 0, "thinking"),
        {"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 238, "estimated_tokens_delta": 188},
        {"type": "stream_event", "event": {
            "type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ""}}},
        {"type": "user", "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "t1", "is_error": True,
            "content": [{"type": "text", "text": "Exit code 1\nboom"}]}]}},
        {"type": "system", "subtype": "status", "status": None, "permissionMode": "acceptEdits"},
        {"type": "system", "subtype": "status", "status": "compacting"},
        {"type": "control_response", "response": {
            "subtype": "success", "request_id": "grokbot-set_permission_mode-1a2b", "response": {"mode": "auto"}}},
        {"type": "control_response", "response": {
            "subtype": "error", "request_id": "grokbot-set_permission_mode-3c4d", "error": "nope"}},
        {"type": "control_response", "response": {
            "subtype": "success", "request_id": "grokbot-interrupt-5e6f", "response": {}}},
    ]
    events = [event for message in messages for event in parser.parse(message)]
    assert events == [
        claude.Ready("s1", "m", "none", "auto"),
        claude.Thinking(),
        claude.Thinking(tokens=238),
        claude.ToolResult("t1", False, "Exit code 1\nboom"),
        claude.Status(""),
        claude.ModeChanged("acceptEdits"),
        claude.Status("compacting"),
        claude.ModeChanged("auto"),
        claude.ControlError("nope", "set_permission_mode"),
    ]


def test_helper_agent_steps_are_nested_and_its_words_are_not_spoken():
    parser = claude.StreamParser()
    [call] = parser.parse({"type": "assistant", "parent_tool_use_id": "task1", "message": {"content": [
        {"type": "tool_use", "id": "t9", "name": "Read", "input": {"file_path": "a/b.py"}}]}})
    assert call == claude.ToolCall("Read", "b.py", "t9", nested=True)
    assert call.input == {"file_path": "a/b.py"}
    parser.parse(block("content_block_start", 0))
    assert parser.parse({**delta(0, "helper chatter"), "parent_tool_use_id": "task1"}) == []


def test_plan_env_hides_api_billing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("SOMETHING_ELSE", "kept")
    env = claude.plan_env()
    assert "ANTHROPIC_API_KEY" not in env and "CLAUDE_CODE_USE_BEDROCK" not in env
    assert env["SOMETHING_ELSE"] == "kept"
