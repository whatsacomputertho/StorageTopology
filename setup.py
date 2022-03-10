import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='StorageTopology',  
     version='1.0',
     scripts=['./StorageTopology.py'] ,
     author="Ethan Balcik",
     author_email="ethanbalcik@ibm.com",
     description="A scientific computation library for performance analysis of erasure coding and/or replication in a distributed data storage environment.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/whatsacomputertho/StorageTopology",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )