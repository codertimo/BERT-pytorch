from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys

__version__ = "0.0.1a1"

with open("requirements.txt") as f:
    require_packages = [line[:-1] for line in f]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != __version__:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, __version__
            )
            sys.exit(info)


setup(
    name="bert_pytorch",
    version=__version__,
    author='Junseong Kim',
    author_email='codertimo@gmail.com',
    packages=find_packages(),
    install_requires=require_packages,
    url="https://github.com/codertimo/BERT-pytorch",
    description="Google AI 2018 BERT pytorch implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'bert = bert_pytorch.train:train',
            'bert-dataset = bert_pytorch.build_dataset:build',
            'bert-vocab = bert_pytorch.build_vocab:build',
        ]
    },
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
