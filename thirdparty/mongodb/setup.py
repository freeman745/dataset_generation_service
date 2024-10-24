from setuptools import setup, find_packages

package_name = "mongodb"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "setuptools",
        "pymongo",
        "python-git",
        "gitdb",
        "gitpython",
        "numpy",
        "opencv-contrib-python",
        "matplotlib",
        # "open3d",
        # "pyntcloud",
        "pythreejs",
        "pyaml",
        "blosc",
        "ipywidgets",
        "plotly",
        "redis",
    ],
    package_data={"": ["**/*.yaml", "**/*.yml"]},
    include_package_data=True,
    zip_safe=True,
)
