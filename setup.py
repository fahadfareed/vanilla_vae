#from setuptools import find_packages, setup

#setup(
#   name='src',
#   packages=find_packages(where="src"),
#   package_dir={"": "src"},
#   version='1.0',
#   description='An implementation of the Variational Autoencoder',
#   author='Fahad Fareed',
#   license='',
#    entry_points='''
#        [console_scripts]
#        vanilla_vae=vanilla_vae.cli:entry_point
#    ''',
#)

from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='An implementation of the Variational Autoencoder',
    author='Fahad Fareed',
    license='',
     entry_points='''
        [console_scripts]
        src=src.cli:entry_point
    ''',
)