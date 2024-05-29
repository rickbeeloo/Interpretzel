from setuptools import setup, find_packages

setup(
    name="interpretzel",
    version = "0.0.1",
    packages=find_packages(),
    install_requires = [
        "vllm", "aqlm", "flash-attn"
    ],

    entry_points = {
        'console_scripts': [
            'iso_desc = interpretzel.main:pretzel_iso_gen',
            'iso_pred = interpretzel.main:pretzel_iso_pred',
        ],
    }
    

)