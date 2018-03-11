import os
from setuptools import find_packages, setup

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='opencv-rolling-ball',
    packages=find_packages(),
    version='1.0',
    description='Fully Ported to Python from ImageJ\'s Background Subtractor.'
                'Only works for 8-bit greyscale images currently. ',
    license='MIT License',
    author='Maksym Balatsko',
    author_email='mbalatsko@gmail.com',
    url='https://github.com/mbalatsko/opencv-rolling-ball',
    download_url='https://github.com/mbalatsko/opencv-rolling-ball/archive/1.0.tar.gz',
    install_requires=[
          'opencv-python',
          'numpy'
      ],
    keywords=['opencv', 'background subtraction', 'rolling ball algorithm'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
)