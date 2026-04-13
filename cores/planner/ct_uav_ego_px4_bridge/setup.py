from setuptools import setup
import os
from glob import glob

package_name = 'ego_px4_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    # package_dir={'': 'src'},
    
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'script'), glob('script/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Bridge between Ego Planner and PX4',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge_real = ego_px4_bridge.bridge_real:main',
            'bridge_real_ver2 = ego_px4_bridge.bridge_real_ver2:main',
            'bridge_sim = ego_px4_bridge.bridge_sim:main',
            'bridge_production = ego_px4_bridge.bridge_production:main',
        ],
    },
)
