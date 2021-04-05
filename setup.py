from setuptools import find_packages, setup

setup(
    name="bert",
    version="0.0.1",
    description="package description",
    install_requires=[],
    url="https://github.com/codertimo/python-template",
    author="codertimo",
    author_email="codertimo@gmail.com",
    packages=find_packages(exclude=["tests", "scripts"]),
)
