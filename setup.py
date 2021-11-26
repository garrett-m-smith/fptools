from setuptools import setup

setup(
    name='fptools',
    version='0.1.2',
    description='Tools for first-passage time analyses',
    url='https://github.com/garrett-m-smith/fptools',
    author='Garrett Smith',
    author_email='gasmith@uni-potsdam.de',
    license='GNU GPL v3.0',
    packages=['fptools'],
    install_requires=['numpy', 'scipy', 'tqdm', ],
    test_suite='nose.collector',
    tests_require=['nose']
    )
