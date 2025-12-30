from setuptools import setup, find_packages

setup(
    name="ai-csv-translator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "pandas>=2.0.0",
        "openai>=1.0.0",
        "tqdm>=4.65.0",
        "openpyxl>=3.1.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.7",
) 