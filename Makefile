pip-compile: requirements.in test-requirements.in docs-requirements.in nb-requirements.in dev-requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links test-requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links nb-requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links docs-requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links dev-requirements.in

pip-install:
	pip install -r dev-requirements.txt -e .

format:
	find src tests -name "*.py" | xargs -I % autoflake8 % --in-place
	black src tests
	isort src tests

test-coverage:
	pytest --cov-report=html --cov=src tests
