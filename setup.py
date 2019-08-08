import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='pyforce',
        version="0.1.1",
        description='Computation of force constants of equilbrium geometries on potential energy surfaces.',
        author='Adam Abbott',
        author_email='adabbott@uga.edu',
        url="https://github.com/adabbott/force_constants",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            #'numpy>=1.7','GPy>=1.9','scikit-learn>=0.20','pandas>=0.24','hyperopt>=0.1.1','cclib>=1.6','matplotlib==3.0.3', 'torch>=1.0.1'
            'numpy>=1.7','GPy>=1.9','scikit-learn>=0.20','pandas>=0.24','hyperopt>=0.1.1','cclib>=1.6', 'torch>=1.0.1'
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest-cov',
            ],
        },

        tests_require=[
            'pytest-cov',
        ],

        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )
