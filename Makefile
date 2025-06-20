PDOC_CFG= -c sort_identifiers=False -c latex_math=True -c show_source_code=False

.PHONY: test
test:
	uv run pytest --doctest-modules

.PHONY: live-docs
live-docs:
	uv run pdoc $(PDOC_CFG) --http :8000 poptools

.PHONY: docs
docs:
	rm -rf docs/*
	uv run pdoc $(PDOC_CFG) --html -f -o ./docs poptools
	mv docs/poptools/* docs/
	rmdir docs/poptools

.PHONY: clean
clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find docs -not -name '.gitkeep' -exec rm -rf {} +
