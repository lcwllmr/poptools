SRC_FILES = $(shell find src/poptools -name '*.py' -not -path '*/__pycache__/*')
PDOC_CFG= -c sort_identifiers=False -c latex_math=True -c show_source_code=False
docs: $(SRC_FILES)
	uv run pdoc $(PDOC_CFG) --html -f -o ./docs poptools
	mv docs/poptools/* docs/
	rmdir docs/poptools
