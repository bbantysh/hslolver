import os
from setuptools import setup


def read(filename: str):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name="hsolver",
    version="0.2",
    author="Boris Bantysh",
    author_email="bbantysh60000@gmail.com",
    description="The tool for solving quantum state evolution under periodic hamiltonian",
    license="GPL-3.0",
    keywords="periodic hamiltonian",
    url="https://github.com/bbantysh/hsolver",
    packages=["hsolver"],
    long_description=read("README.md"),
    install_requires=["numpy", "scipy", "pytest"]
)
