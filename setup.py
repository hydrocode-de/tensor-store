from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()
    

def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


def version():
    loc = dict()
    with open('tensorage/__version__.py') as f:
        exec(f.read(), loc, loc)
        return loc['__version__']
    

setup(
    name='tensorage',
    version=version(),
    description='Storing tensors in a supabase backend',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Mirko Mälicke',
    author_email='mirko@hydrocode.de',
    install_requires=requirements(),
    packages=find_packages(),
)
