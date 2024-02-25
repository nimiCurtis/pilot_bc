from setuptools import setup, find_packages

setup(
    name='pilot_train',
    version='0.1',
    packages=find_packages(),
    # entry_points={
    #     'console_scripts': [
    #         'bagtool=bagtool.bagtool_main:main',  # 'command=package.module:function'
    #     ],
    # },
    install_requires=[
        # List your package dependencies here, for example:
        # 'numpy',
        # 'pandas',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A simple bagtool command line tool',
    license='MIT',
)