from setuptools import setup, find_packages
from typing import List


def get_install_requires(file_path: str = "requirements.txt") -> List[str]:
    with open(file_path) as f:
        install_requires = f.read().splitlines()
    return install_requires

setup(
    name="text_augmentation",
    version="1.0.0",
    url="https://github.com/Jaesu26/text-augmentation",
    license="MIT",
    author="Jaesu Han",
    author_email="gkswotn9753@gmail.com",
    description="Text augmentation using albumentations package",
    packages=find_packages(),
    long_description=open("README.md", encoding="utf-8").read(),
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=get_install_requires(),
)
