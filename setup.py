import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='xypy',
      version='0.0.6',
      description='A simulation framework for supervised learning data. The functionalities are specifically\
                   designed to let the user a maximum degrees of freedom, to ultimately fulfill the research purpose.\
                   Furthermore, feature importances of the simulation can be created on a local and a global level. \
                   This is particular interesting, for instance, to benchmark feature selection algorithms.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      py_modules=['xypy'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        "Operating System :: OS Independent",
        'License :: OSI Approved :: MIT License  ',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence  ',
      ],
      keywords='simulation machine learning supervised feature selection',
      url='http://github.com/andrebleier/XyPy',
      author='André Bleier',
      author_email='andrebleier@live.de',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[
         'scipy', 'numpy', 'pandas', 'seaborn', 'matplotlib', 'statsmodels', 'sklearn'
      ]
)