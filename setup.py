import os
from setuptools import setup


def read(filename: str):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name='iloptics',
    version='1.0',
    author='Boris Bantysh',
    author_email='bbantysh60000@gmail.com',
    description='The package for simulating and training integrated linear optical devices',
    license='GPL-3.0',
    keywords='linear optics training',
    url='https://github.com/PQCLab/ILOptics',
    packages=['iloptics'],
    long_description=read('README.md'),
    install_requires=['numpy', 'scipy', 'tqdm']
)
