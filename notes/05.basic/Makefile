basic.pdf: basic.tex results/Table1.tex results/Table2.tex results/Table3.tex \
	   results/Table4.tex results/Table5.tex results/Table6.tex

	  # run twice to resolve references
	  pdflatex -output-directory ../tmp basic.tex
	  pdflatex -output-directory ../tmp basic.tex
	  mv ../tmp/basic.pdf ..
