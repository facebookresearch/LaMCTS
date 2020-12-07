import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
     name = 'LA-MCTS',  
     version = '0.1',
     author = "Linnan Wang",
     author_email = "wangnan318@gmail.com",
     description = "",
     long_description = long_description,
     long_description_content_type = "text/markdown",
     url = "https://github.com/facebookresearch/LaMCTS",
     packages = ["lamcts"],
     install_requires=required,
     include_package_data = True,
     classifiers = [
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ]
 )
