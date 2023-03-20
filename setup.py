from setuptools import find_packages, setup


setup(
    name="textmentations",
    version="0.0.2",
    description="A Python library for augmenting Korean text.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Jaesu Han",
    author_email="gkswotn9753@gmail.com",
    url="https://github.com/Jaesu26/textmentations",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    zip_safe=False,
    include_package_data=True,
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={"tests": ["pytest"]},
    python_requires=">=3.7",
)
