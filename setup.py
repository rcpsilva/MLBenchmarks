from setuptools import setup, find_packages

setup(
    name='MLBenchmarks',
    version='0.1',
    author='Silva, R.',
    author_email='',
    description='An ML Python package',
    long_description='',
    url='https://github.com/rcpsilva/MLBenchmarks',
    packages=find_packages(),
    install_requires=['psutil',
                      'tqdm',
                      'numpy',
                      'scikit-learn',
                      'pandas',
                      'openpyxl',
                      'setuptools'
                      ],
    classifiers=[
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GPL-3.0',
        'Programming Language :: Python :: 3.11',
    ],
)
