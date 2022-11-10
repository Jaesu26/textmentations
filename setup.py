from setuptools import setup, find_packages

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
    install_requires=["albumentations>=1.2.1"],
)
