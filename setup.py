from setuptools import setup
from setuptools import find_packages

setup(name='DeepFrame',
      version='0.0.2',
      description='Deep Learning library in python',
      author='Zhihong Dong',
      author_email='dzhwinter@gmail.com',
      url='http://github.com/dzhwinter/DeepFrame',
      license='MIT',
      install_requires=['theano', 'pyyaml'],
      extras_require= {
          'h5py': ['h5py'],
      },
      packages=find_packages())
