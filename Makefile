# display the makefile usage
help:
	@echo "Tasks:"
	@echo "- build_docs"
	@echo "- build"
	@echo "- sdist"

# run the test harness
#test:
#	python setup.py test

# run coverage
#coverage:
#	python setup.py test --addopts "--cov trendy --cov-report html --cov-report term"

# build the docs
build_docs:
	python setup.py build_sphinx

# build the package
build:
	python setup.py build

# create a common source distribution
sdist:
	python setup.py sdist

# install the package
install:
	python setup.py install
