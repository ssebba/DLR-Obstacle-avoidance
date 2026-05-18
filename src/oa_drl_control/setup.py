from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'oa_drl_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'model.config']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world') + glob('worlds/*.obj')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seba',
    maintainer_email='seba@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'filter_lidar = oa_drl_control.lidar_data_filter:main',
            'controller = oa_drl_control.controller:main',
            'trainer = oa_drl_control.trainer:main',
            'respawner = oa_drl_control.respawner:main'
        ],
    },
)
