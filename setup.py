from setuptools import setup, find_packages

setup(
    name='deepcovidnet',
    version='0.1',
    packages=find_packages(exclude=['tests', 'data', 'runs', 'models'])
    #can add entrypoints here
)
