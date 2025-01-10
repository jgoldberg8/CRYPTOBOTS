from setuptools import setup, find_packages


setup(
    name='NNBOT',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'packages': [
            'TimeToPeak=TimeToPeak',
            'PeakMarketCap=PeakMarketCap',
            'utils=utils',
            'before30=before30'
        ]
    }
)