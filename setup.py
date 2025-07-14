from setuptools import setup, find_packages

setup(
    name="lqg_simulation",
    version="0.1.0",
    description="Research-grade Loop Quantum Gravity simulation toolkit (spin networks, spinfoams, quantum, visualization, plugins)",
    author="natarajanphysicist and contributors",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "networkx",
        "sympy",
        "ipywidgets",
        # Optional: "qiskit", "plotly", "pyyaml", etc.
    ],
    include_package_data=True,
    python_requires=">=3.8",
    entry_points={
        # Example CLI entry point (optional)
        # 'console_scripts': [
        #     'lqg-sim=lqg_simulation.__main__:main',
        # ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
