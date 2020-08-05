#nsml: registry.navercorp.com/nsml/airush2020:pytorch1.5
from distutils.core import setup
import os

os.system('apt-get install libglib2.0-0')
setup(
    name="Spam classification - AI Rush baseline",
    version="1",
    install_requires=['efficientnet_pytorch',
        'xgboost'
    ]
)
