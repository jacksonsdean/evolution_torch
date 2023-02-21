import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='evolution_torch',
    version='0.0.8',
    author='Jackson Dean',
    author_email='jackson@downbeat.games',
    description='Evolutionary algorithm implementation in PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jacksonsdean/evolution_torch',
    project_urls = {
        "Bug Tracker": "https://github.com/jacksonsdean/evolution_torch/issues"
    },
    license='MIT',
    packages=['evolution_torch'],
    install_requires=['torch', 'numpy', 'matplotlib', 'scipy', 'scikit-image', 'functorch', 'tqdm', 'torchvision', 'piq', 'imageio','scikits-bootstrap', 'networkx'],
)