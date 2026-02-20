import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='gridmlip',  
     version='0.2.1',
     py_modules = ["gridmlip"],
     install_requires = [
                         "pandas",
                         "numpy",
                         "scipy",
                         "ase",
                         "tqdm",
                         "joblib",
                         ],
     author="Artem Dembitskiy",
     author_email="art.dembitskiy@gmail.com",
     description="Grid-based method for calculating percolation barriers of mobile species using machine learning interatomic potentials",
     key_words = ['percolation-barrier', 'UMLIP', 'ionic conductivity', 'diffusion'],
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/dembart/gridmlip",
     package_data={"gridmlip": ["*.rst", '*.md'], 
                    #'tests':['*'], 
                    },
     classifiers=[
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "Programming Language :: Python :: 3.10",
         "Programming Language :: Python :: 3.11",
         "Programming Language :: Python :: 3.12",
         "Programming Language :: Python :: 3.13",
         "Programming Language :: Python :: 3.14",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    include_package_data=True,
    packages=setuptools.find_packages(),
)