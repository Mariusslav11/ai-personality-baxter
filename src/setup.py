from setuptools import find_packages, setup
import os
import sys

# Ensure venv is activated before running
venv_path = os.path.join(os.path.dirname(__file__), "venv", "bin", "python3")
if os.path.exists(venv_path):
    sys.executable = venv_path

package_name = 'baxter_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marius',
    maintainer_email='marius@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chatbot_run = baxter_pkg.chatbot_run:main',
        ],
    },
)
