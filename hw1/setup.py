
from setuptools import find_packages
from setuptools import setup 

with open( 'requirements.txt' ) as fhandle :
    _requirements = fhandle.read().splitlines()

packages = find_packages()

setup(
    name                    = 'cs294-imitation',
    version                 = '0.0.1',
    description             = 'cs294 Homework 1 - imitation learning',
    keywords                = 'rl ai dl',
    packages                = packages,
    install_requires        = _requirements
)