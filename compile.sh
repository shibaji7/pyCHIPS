rm -rf dist/ build/ pyCHIPS.egg-info/

find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
isort -rc -sl chips
autoflake --remove-all-unused-imports -i -r chips
isort -rc -m 3 chips
black chips

python setup.py sdist bdist_wheel
