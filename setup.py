from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n", " ") for i in requirements]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
        
    return requirements
        
setup(
    name= "Diamond Price Prediction",
    version= '0.0.1',
    author = "Anwesha Chowdhury",
    author_email= "achowdhury1211@gmail.com",
    requires_install = get_requirements('requirements.txt'),
    packages = find_packages()
)
