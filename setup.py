from setuptools import setup, find_packages

setup(
    name='rfi_toolbox',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'torch>=1.9.0',
        'bokeh',
        'scikit-learn',
        'scipy',
        'albumentations'
    ],
    entry_points={
        'console_scripts': [
            'generate_rfi_dataset=rfi_toolbox.scripts.generate_dataset:main',
            'train_rfi_model=rfi_toolbox.scripts.train_model:main',
            'evaluate_rfi_model=rfi_toolbox.scripts.evaluate_model:main',
            'visualize_rfi_data=rfi_toolbox.visualization.visualize:main',
            'normalize_rfi_data=rfi_toolbox.scripts.normalize_rfi_data:main',
        ],
    },
)