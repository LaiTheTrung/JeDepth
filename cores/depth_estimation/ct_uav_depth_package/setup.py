from setuptools import find_packages, setup
import os
from glob import glob

def get_folder_tree(root_path):
    tree = []
    for dirpath, dirnames, _ in os.walk(root_path):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if '__pycache__' in full_path:
                continue
            tree.append(full_path)
    return tree

package_name = 'ct_uav_depth_package'
submodules_omnistereo = ['models/omnistereo'] + get_folder_tree('models/omnistereo')
submodules_hitnet = ['models/hitnet'] + get_folder_tree('models/hitnet')
submodules_fastacv = ['models/fast_acv_plus'] + get_folder_tree('models/fast_acv_plus')
submodules_rfdert = ['models/rfdert_seg'] + get_folder_tree('models/rfdert_seg')
packages = [package_name] + submodules_omnistereo + submodules_hitnet + submodules_fastacv + submodules_rfdert + ["config"]
setup(
    name=package_name,
    version='0.0.0',
    packages=packages,
    package_data={
        'config': ['*'],
        'models/hitnet': ['nvdsinfer_so/*', 'config_infer_hitnet.txt'],
        'models/omnistereo': ['config/*'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ctuav',
    maintainer_email='ctuav@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'demo_one = ct_uav_depth_package.demo_one:main',
            'demo_two = ct_uav_depth_package.demo_two:main',
            'demo_three = ct_uav_depth_package.demo_three:main',
        ],
    },
)
