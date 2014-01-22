tmpdir := tmp

FILES := p1.pdf s1.pdf p2.pdf s2.pdf p3.pdf

all: $(FILES)

$(tmpdir):
	mkdir tmp

p1.pdf: | $(tmpdir)
	make -C ex1

p2.pdf: | $(tmpdir)
	make -C ex2

p3.pdf: | $(tmpdir)
	make -C ex3

s1.pdf: | $(tmpdir)
	make -C ex1

s2.pdf: | $(tmpdir)
	make -C ex1

clean:
	rm -rf tmp $(FILES)
