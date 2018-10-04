from setuptools import setup, find_packages

setup(
    name="it3105",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "spec = it3105.command_line:run",
        ]
    },
    install_requires=[
        'logzero',
        'tensorflow==1.11.0',
        'cerberus==1.2',
        'toml==0.9.6 ',
        'docopt==0.6.2',
        'PyQt5==5.11.2',
        'pyqtgraph==0.10.0',
        'scipy==1.1.0',
        'matplotlib==3.0.0'
    ],
)
