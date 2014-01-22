all: p2.pdf s2.pdf

p2.pdf: p2.tex ctest3.pdf
	# twice to handle references
	pdflatex -output-directory ../tmp p2.tex
	pdflatex -output-directory ../tmp p2.tex
	mv ../tmp/p2.pdf ..

s2.pdf: s2.tex
	# twice to handle references
	pdflatex -output-directory ../tmp s2.tex
	pdflatex -output-directory ../tmp s2.tex
	mv ../tmp/s2.pdf ..
