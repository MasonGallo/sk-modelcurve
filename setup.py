import sys

try:
    from setuptools import setup, Command
except ImportError:
    # no setuptools installed
    from distutils.core import setup, Command

if sys.version_info[0] >= 3:
    sys.exit('Sorry! Python 3+ not yet supported.')

min_numpy_ver = '1.7.0'
min_sklearn_ver = '0.16.0'
min_matplotlib_ver = '1.5.0'

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='sk_modelcurves',
      version='0.1',
      description='A wrapper for easy creation of Learning and Validation Curves',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
      ],
      keywords='sk_modelcurves learning curves validation curves',
      url='http://github.com/masongallo/sk-modelcurve',
      author='Mason Gallo',
      author_email='masongallo@gatech.edu',
      license='MIT',
      packages=['sk_modelcurves'],
      install_requires=[
          'matplotlib',
          'numpy',
          'scikit-learn',
      ],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

