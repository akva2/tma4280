poisson-fdm.pdf: poisson-fdm.tex FiniteDifference.pdf Poisson1D_Domain.pdf \
  		 Poisson1D_Grid.pdf Poisson2D_Domain.pdf Poisson2D_Grid.pdf \
		 FivePointStencil.pdf NaturalOrdering.pdf \
		 Poisson2D_System.pdf BandedMatrix.pdf Poisson2D_ntimesnGrid.pdf \
		 Poisson2D_aspect1.pdf Poisson2D_aspect2.pdf
	  	 # run twice to resolve references
	  	 pdflatex -output-directory ../tmp poisson-fdm.tex
	  	 pdflatex -output-directory ../tmp poisson-fdm.tex
	  	 mv ../tmp/poisson-fdm.pdf ..
