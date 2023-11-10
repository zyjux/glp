from setuptools import setup, find_packages
import os

# For guidance on setuptools best practices visit
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
project_name = "glp"
version = "0.1.0"
package_description = "<Provide short description of package>"
url = "https://github.com/zyjux/" + project_name
# Classifiers listed at https://pypi.org/classifiers/
classifiers = ["Programming Language :: Python :: 3"]
setup(
    name=project_name,
    version=version,
    description=package_description,
    url=url,
    author="Lander Ver Hoef",
    license="CC0 1.0",
    classifiers=classifiers,
    packages=["glp_ri"],  # Specify the package name here
    package_dir={"glp_ri": "glp_ri"},  # Specify the directory mapping
)
