import setuptools

setuptools.setup(     
     name="loca",     
     version="0.0.1",
     python_requires=">=3.9",   
     packages=["loca"],
     package_dir={"loca": "models"},
     install_requires=["torch", "torchvision"],
)