import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="fluxcones",
    version="1.0.0",
    author="Frederik Wieder",
    description=("The fluxcones package"),
    url="https://github.com/fwieder/fluxcones",
    packages=["fluxcones"],
    long_description=read("Readme.md"),
    install_requires=[
        "numpy",
        "mip",
        "efmtool",
        "scipy",
        "cobra",
        "pycddlib",
        "tqdm"
    ],
)