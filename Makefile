PDOC_CFG= -c sort_identifiers=False -c latex_math=True -c show_source_code=False

.PHONY: test
test:
	uv run -m pytest --cov=poptools --doctest-modules

.PHONY: live-docs
live-docs:
	uv run pdoc $(PDOC_CFG) --http :8000 poptools

.PHONY: docs
docs:
	rm -rf docs/*
	uv run pdoc $(PDOC_CFG) --html -f -o ./docs poptools
	mv docs/poptools/* docs/
	rmdir docs/poptools

.PHONY: badges
badges:
	mkdir -p docs/badges
	COVERAGE=$$(uvx coverage report | grep '^TOTAL' | awk '{print $$NF}' | sed 's/%/%25/'); \
	curl -o docs/badges/coverage.svg https://img.shields.io/badge/coverage-$$COVERAGE-brightgreen

.PHONY: clean
clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	rm -rf ./docs/
