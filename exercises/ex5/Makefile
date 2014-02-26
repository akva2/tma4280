all: p5.pdf

p5.pdf: p5.tex
	# twice to handle references
	pdflatex -output-directory ../tmp p5.tex
	pdflatex -output-directory ../tmp p5.tex
	mv ../tmp/p5.pdf ..
