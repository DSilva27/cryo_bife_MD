
"""Create instructions to build the simSPI package."""

import setuptools

requirements = []

setuptools.setup(
    name="cryoBIFE",
    maintainer=[
        "David Silva-Sánchez",
        "Julian David Giraldo-Barreto",
        "Erik Henning Thiede",
        "Pilar Cossio",
        "Sonya Hanson",
    ],
    version="0.0.1",
    maintainer_email=[
        "david.silva@yale.edu",
    ],
    description="Path optimization using CryoBIFE",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DSilva27/cryo_bife_MD.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)