all: p4.pdf

p4.pdf: p4.tex
	# twice to handle references
	pdflatex -output-directory ../tmp p4.tex
	pdflatex -output-directory ../tmp p4.tex
	mv ../tmp/p4.pdf ..
