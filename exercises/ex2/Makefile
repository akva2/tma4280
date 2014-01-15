all: p2.pdf

p2.pdf: p2.tex ctest3.pdf
	# twice to handle references
	pdflatex -output-directory ../tmp p2.tex
	pdflatex -output-directory ../tmp p2.tex
	mv ../tmp/p2.pdf ..
