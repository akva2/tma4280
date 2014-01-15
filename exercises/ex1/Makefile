all: p1.pdf s1.pdf

p1.pdf: p1.tex FloatingPointRep.pdf
	pdflatex -output-directory ../tmp p1.tex
	mv ../tmp/p1.pdf ..

s1.pdf: s1.tex
	pdflatex -output-directory ../tmp s1.tex
	mv ../tmp/s1.pdf ..
