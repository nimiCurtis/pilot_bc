from setuptools import setup, find_packages

# Read the content of your requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pilot_bc',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
)