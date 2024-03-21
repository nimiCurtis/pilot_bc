from setuptools import setup, find_packages

setup(
    name='pilot_bc',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list your project's dependencies here, for example:
        # 'numpy',
        # 'pandas',
    ],
    # If your packages are not in the root directory, use the package_dir argument
    # package_dir={'': 'src'},
    # You can specify package data, scripts, and more. See the setuptools documentation for details.
)