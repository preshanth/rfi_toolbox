# rfi_toolbox/setup.py
from setuptools import setup, find_packages

setup(
    name='rfi_toolbox',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'torch',
        'bokeh'
    ],
    entry_points={
        'console_scripts': [
            'generate_rfi_dataset=rfi_toolbox.scripts.generate_dataset:main',
        ],
    },
)
