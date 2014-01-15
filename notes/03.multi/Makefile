multi_notes.pdf: 2Dmesh.pdf 2Dtoroid.pdf bus_organization.pdf ccNUMA.pdf crossbar.pdf hypercube.pdf \
	   	 message_passing_model.pdf MIMD.pdf multiprocess1.pdf multiprocess2.pdf P5.pdf \
	   	 multistage_interconnect.pdf multi_notes.tex node.pdf NUMA.pdf scalability.pdf SMP.pdf
	# run twice to resolve references
	pdflatex -output-directory ../tmp multi_notes.tex
	pdflatex -output-directory ../tmp multi_notes.tex
	mv ../tmp/multi_notes.pdf ..
