.PHONY: fmt lint typecheck test test-quantum test-all test-cov clean


# ── Formatting & linting ──────────────────────────────────────────────────────

fmt:
	ruff format src/ tests/ examples/ benchmarks/

fmt-check:
	ruff format --check src/ tests/ examples/ benchmarks/

lint:
	ruff check src/ tests/ examples/ benchmarks/

lint-fix:
	ruff check --fix src/ tests/ examples/ benchmarks/

typecheck:
	mypy src/qportfolio

# Run fmt + lint + typecheck
check: fmt-check lint typecheck

# ── Testing ───────────────────────────────────────────────────────────────────

# Fast: unit + regression only (no quantum, no network)
test:
	pytest -q -m "not quantum and not slow and not integration"

# Quantum tests (requires qiskit + qiskit-aer)
test-quantum:
	pytest -q -m "quantum and not julia" tests/quantum/

# Julia-dependent tests (requires julia on PATH)
test-julia:
	pytest -q -m "julia" tests/quantum/

# Integration tests (hits network / disk)
test-integration:
	pytest -q -m "integration"

# Everything except slow
test-all:
	pytest -q -m "not slow"

# With coverage report
test-cov:
	pytest -q -m "not quantum and not slow and not integration" \
	    --cov=src/qportfolio --cov-report=term-missing --cov-report=html

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf .pytest_cache htmlcov .coverage dist build
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
