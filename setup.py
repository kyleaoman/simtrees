from setuptools import setup

setup(
    name='simtrees',
    version='1.0',
    description='Interface to John Helly merger tree files.',
    url='',
    author='Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='GNU GPL v3',
    packages=['simtrees'],
    install_requires=['numpy', 'h5py'],
    include_package_data=True,
    zip_safe=False
)
