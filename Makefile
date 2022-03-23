test:
	python3 -m unittest discover --verbose

install:
	pip3 install --requirement requirements.txt
	./make.sh
