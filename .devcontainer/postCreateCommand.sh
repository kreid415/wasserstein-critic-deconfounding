git clone https://github.com/kreid415/scvi-tools.git
cd scvi-tools
git checkout kr_dev
pip install -e .
cd ..

pre-commit install

pip install -e .

pip install ipywidgets --upgrade
