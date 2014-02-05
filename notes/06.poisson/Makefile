poisson.pdf: dd_grid.pdf HeatFlux.pdf poisson.tex
	  # run twice to resolve references
	  pdflatex -output-directory ../tmp poisson.tex
	  pdflatex -output-directory ../tmp poisson.tex
	  mv ../tmp/poisson.pdf ..
