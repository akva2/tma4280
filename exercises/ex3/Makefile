all: p3.pdf

p3.pdf: p3.tex
	# twice to handle references
	pdflatex -output-directory ../tmp p3.tex
	mv ../tmp/p3.pdf ..
