# Who is Baxter
Baxter is a charming AI chatbot companion designed with 4 personalities and real-time info retrieval capabilities. Its response delivery latency is less than 2 seconds for 90% of queries, leveraging OpenAI's streaming TTS service. He also has an audio visualizer, synced to the output of the sound file created from TTS. This is used on the GUI to accompany the companion's personality.

# Baxter Chatbot Setup Guide

# Prerequisites
--Windows 10/11 with WSL2 installed

--Ubuntu 22.04 LTS (recommended) as your WSL2 distro

--ROS2 Jazzy installed and sourced

--Python 3.12 installed in WSL2

--VSCode with the Remote - WSL extension

# Setup Instructions
-Clone the repository
Create and activate a virtual environment (Python 3.12)

--Check Python version
python3.12 --version  

--Create virtual environment using Python 3.12
python3.12 -m venv venv

--Activate the virtual environment
source venv/bin/activate

--Upgrade pip (inside the venv)
python -m pip install --upgrade pip

--Install dependencies
pip install -r requirements.txt

--Download and source ROS2 Jazzy
Once downloaded source like this:
source /opt/ros/jazzy/setup.bash

# Running the Chatbot
Make sure you are in the baxter_pkg package, have ROS2 sourced and dependencies installed
Do this inside of the venv, then run the chatbot using:

python3 chatbot_node.py
