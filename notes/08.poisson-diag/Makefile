poisson-diag.pdf: poisson-diag.tex Domain2D_ntimesn.pdf FiniteDifferenceGrid.pdf \
		  PeriodicFunction.pdf
	  	 # run twice to resolve references
	  	 pdflatex -output-directory ../tmp poisson-diag.tex
	  	 pdflatex -output-directory ../tmp poisson-diag.tex
	  	 pdflatex -output-directory ../tmp poisson-diag.tex
	  	 mv ../tmp/poisson-diag.pdf ..
