from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vision_cenital'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Instalar archivos de recursos (YAMLs)
        (os.path.join('share', package_name, 'resource'), glob('resource/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Felipe',
    description='Visión cenital y control para CargaBot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Expone tu nodo principal al comando 'ros2 run'
            'coordinator_node = vision_cenital.overhead_coordinator_node:main',
            'virtual_cargabot_node = vision_cenital.virtual_cargabot_node:main',
        ],
    },
)