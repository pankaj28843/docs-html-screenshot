---
applyTo:
  - "**/tests/**"
---

# Test Instructions (docs-html-screenshot)

## General Rules

- Use `@pytest.mark.asyncio` for all async test functions
- Test method names start with `test_` describing behavior (e.g., `test_discover_html_files_returns_sorted_list`)
- No docstrings in test classes/methods—clear method names suffice
- Avoid verbose comments; test code should be self-documenting
- Use fixtures over ad-hoc setup to reduce duplication

## Test Behavior, Not Implementation

```python
# GOOD: Tests what the function does
def test_discover_html_files_finds_nested_files(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "sub" / "page.html").mkdir(parents=True)
    (tmp_path / "sub" / "page.html").write_text("<html></html>")
    
    result = discover_html_files(tmp_path)
    assert len(result) == 2

# BAD: Tests how it does it
def test_uses_rglob_method():
    mock_path = Mock()
    discover_html_files(mock_path)
    mock_path.rglob.assert_called_once()  # Testing implementation
```

## Mock Only External Boundaries

```python
# GOOD: Mock external browser calls
with patch.object(context, "new_page") as mock_page:
    mock_page.return_value = async_mock_page
    result = await process_file(context, config, ...)

# BAD: Mock internal methods
with patch.object(service, "_internal_helper"):
    # Fragile, breaks on refactoring
```

## Test Organization

```
tests/
├── unit/  # Fast, isolated tests (NO external dependencies)
│   ├── test_cli_helpers.py   # Helper function tests
│   └── test_*.py
├── integration/  # May use real browser (marked slow)
│   └── test_*.py
└── conftest.py  # Shared fixtures
```

## Validation Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src/docs_html_screenshot --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_cli_helpers.py -v

# Run tests matching pattern
uv run pytest tests/ -k "test_discover" -v
```

## Anti-Patterns

- **No fake tests** - Don't assert that constants equal literals
- **No mocking framework internals** - Only mock true external dependencies
- **No tests requiring real browsers** in unit tests - Mark as integration
- **No tests with network calls** in unit tests - Mock HTTP clients
