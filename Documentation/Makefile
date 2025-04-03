

all: $(patsubst %.tex,%.pdf,$(wildcard *.tex))

%.pdf: %.tex
	@latexmk $^

.PHONY: clean
clean: $(wildcard *.tex)
	@latexmk -C -bibtex $^
