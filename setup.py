import setuptools
with open('ReadME.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

__version__ = '0.0.0'
REPO_NAME = 'End-to-End-Image-Captioning-With-Attention'
AUTHOR_USER_NAME = 'Asatheesh6561'
SRC_REPO = 'imageCaptioningWithAttention'
AUTHOR_EMAIL = 'anirudh.satheesh@gmail.com'

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='End to End project for image captioning with attention.',
    long_description=long_description,
    long_description_content='text/markdown',
    url=f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}',
    project_urls = {
        'Bug Tracker': f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues'
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src')
)