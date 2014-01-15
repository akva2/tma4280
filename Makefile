all:
	make -C notes
	make -C slides

clean:
	make -C notes clean
	make -C slides clean
