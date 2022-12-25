from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="titanic",
    version="0.1",
    url="https://github.com/danielKatagiri/basic-mlops",
    description="Package to create a predictor for the Titanic dataset",
    packages=find_packages(exclude=["train*"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pre-commit >= 2.12",
        ],
    },
)
