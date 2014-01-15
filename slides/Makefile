tmpdir := tmp

all: curriculum.pdf trends.pdf thetool.pdf crash.pdf multi.pdf

$(tmpdir):
	mkdir tmp

curriculum.pdf: | $(tmpdir)
	make -C 00.curriculum

trends.pdf: | $(tmpdir)
	make -C 01.trends

thetool.pdf: | $(tmpdir)
	make -C 02.thetool

crash.pdf: | $(tmpdir)
	make -C 03.crash

multi.pdf: | $(tmpdir)
	make -C 04.multi

clean:
	rm -rf tmp curriculum.pdf trends.pdf thetool.pdf crash.pdf multi.pdf
