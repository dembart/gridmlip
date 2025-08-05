import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='gridmlip',  
     version='0.1.1',
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
     description="Grid-based method for calculating the percolation barrier of mobile species using machine learning interatomic potentials",
     key_words = ['percolation-barrier', 'umlip', 'conductivity', 'diffusion'],
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/dembart/gridmlip",
     package_data={"gridmlip": ["*.rst", '*.md'], 
                    #'tests':['*'], 
                    },
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    include_package_data=True,
    packages=setuptools.find_packages(),
)