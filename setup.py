from setuptools import setup, find_packages

setup(
    name='pytorch_3d_learning',
    version='0.1',
    packages=find_packages(),  # Automatically discover packages
    author='Lucas Krenmayr',
    description='3D deep learning experiments with PyTorch',
    long_description_content_type='text/markdown',
    url='https://github.com/LucasKre/pytorch_3d_learning',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        # Add your project dependencies here
    ],
)