from setuptools import setup, find_packages

setup(
    name="genai-lib",
    version="0.1.0",
    description="A lightweight Generative AI library with LLM and prompting tools",
    author="Suraj Patra",
    author_email="suraj.patra@involead.com",
    url="https://github.com/yourusername/genai-lib",
    packages=find_packages(),
    install_requires=[
        "requests>=2.20.0",
    ],
    python_requires=">=3.8",
)
