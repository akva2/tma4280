all:
	make -C notes
	make -C slides
	make -C exercises

clean:
	make -C notes clean
	make -C slides clean
	make -C exercises clean
