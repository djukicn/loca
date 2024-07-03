import setuptools
from setuptools import find_packages

setuptools.setup(     
     name="loca",     
     version="0.0.1",
     python_requires=">=3.9",   
     packages=find_packages(),
     install_requires=["torch"],
)