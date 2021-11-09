from setuptools import find_packages, setup

setup(
    name='vanilla_vae',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version='1.0',
    description='An implementation of the Variational Autoencoder',
    author='Fahad Fareed',
    license='',
     entry_points='''
        [console_scripts]
        vanilla_vae=vanilla_vae.cli:entry_point
    ''',
)
