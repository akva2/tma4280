all: p6.pdf

p6.pdf: p6.tex matrix_blocktranspose.pdf
	# twice to handle references
	pdflatex -output-directory ../tmp p6.tex
	pdflatex -output-directory ../tmp p6.tex
	mv ../tmp/p6.pdf ..
