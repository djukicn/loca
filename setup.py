import setuptools
from setuptools import find_packages

setuptools.setup(     
     name="loca",     
     version="0.0.1",
     python_requires=">=3.9",   
     packages=find_packages(),
     package_dir={"loca": "models"},
     install_requires=["torch", "torchvision"],
)