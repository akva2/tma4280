tmpdir := tmp

all: p1.pdf s1.pdf

$(tmpdir):
	mkdir tmp

p1.pdf: | $(tmpdir)
	make -C ex1

s1.pdf: | $(tmpdir)
	make -C ex1

clean:
	rm -rf tmp p1.pdf s1.pdf
