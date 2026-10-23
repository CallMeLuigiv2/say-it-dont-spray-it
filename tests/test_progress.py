from grokbot.claude import ToolCall, ToolResult
from grokbot.progress import Progress, activity, clock, describe_command


def test_activity_in_spoken_words():
    assert activity("Read", {"file_path": r"C:\proj\src\app.py"}) == "reading app.py"
    assert activity("Edit", {"file_path": "/a/b/loader.py"}) == "editing loader.py"
    assert activity("Write", {"file_path": "notes.md"}) == "writing notes.md"
    assert activity("Bash", {"command": "pytest", "description": "Run the tests"}) == "running the tests"
    assert activity("PowerShell", {"description": "List files in the folder"}) == "listing files in the folder"
    assert activity("Bash", {"command": "ls"}) == "running a command"
    assert activity("Grep", {"pattern": "def main"}) == "searching the code"
    assert activity("WebFetch", {"url": "https://www.example.com/docs"}) == "reading example.com"
    assert activity("Task", {"description": "Find the config loader"}) == "sending a helper to find the config loader"
    assert activity("mcp__github__create_issue", {}) == "using create issue"


def test_command_descriptions_become_what_im_doing():
    assert describe_command("Create the venv") == "creating the venv"
    assert describe_command("Commit the fix.") == "committing the fix"
    assert describe_command("Checking types") == "checking types"
    assert describe_command("Git status") == "running git status"
    assert describe_command("NPM install") == "running NPM install"


def test_clock():
    assert clock(8.6) == "8s"
    assert clock(65) == "1:05"


def test_a_turn_from_thinking_to_done():
    progress = Progress()
    assert progress.detail(0.0) == ""
    progress.begin(100.0)
    progress.thinking(None, "")
    progress.thinking(1200, "")
    assert progress.phase == "thinking"
    assert progress.detail(105.0) == "5s · 1,200 tokens of thought"
    assert progress.check_in() == "Still thinking."

    progress.tool(ToolCall("Edit", "app.py", "t1", input={"file_path": "src/app.py"}))
    assert progress.phase == "working"
    assert progress.detail(130.0) == "30s · 1 step · editing app.py"
    assert progress.check_in() == "Still working. Editing app.py."
    assert progress.check_in() == "Still on it."  # nothing new to say

    assert progress.result(ToolResult("t1", False, "boom")) == "Edit"
    assert progress.phase == "thinking"
    assert progress.check_in() == "Still working, 1 step in."
    progress.writing()
    assert progress.check_in() == "Still writing."
    assert progress.summary(170.0) == "done in 1:10 · 1 step · 1 failed"

    progress.end()
    assert not progress.active and progress.steps == 0


def test_visible_thoughts_and_helper_steps():
    progress = Progress()
    progress.begin(0.0)
    progress.thinking(None, "")
    progress.thinking(None, "First the loader.\nThen the te")
    progress.thinking(None, "sts.")
    assert progress.detail(1.0) == "1s · “Then the tests.”"
    progress.tool(ToolCall("Read", "b.py", "t2", nested=True, input={"file_path": "a/b.py"}))
    assert progress.doing == "helper: reading b.py"
    assert progress.result(ToolResult("t2", True, "")) is None
