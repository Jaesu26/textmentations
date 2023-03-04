from typing import List

from setuptools import find_packages, setup


def get_install_requires(file_path: str = "requirements.txt") -> List[str]:
    with open(file_path) as f:
        install_requires = f.read().splitlines()
    return install_requires


setup(
    name="text_boost",
    version="1.0.0",
    description="Text augmentation using albumentations",
    long_description=open("README.md", encoding="utf-8").read(),
    author="Jaesu Han",
    author_email="gkswotn9753@gmail.com",
    url="https://github.com/Jaesu26/text-boost",
    license="MIT", 
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=get_install_requires(),
)
