# Who is Baxter?
Baxter is a charming AI chatbot companion designed with 4 personalities and real-time info retrieval capabilities. Its response delivery latency is less than 2 seconds for 90% of queries, leveraging OpenAI's streaming TTS service. He also has an audio visualizer, synced to the output of the sound file created from TTS. This is used on the GUI to accompany the companion's personality.

# Baxter Chatbot Setup Guide

## Prerequisites
-Windows 10/11 with WSL2 installed

-Ubuntu 24.04 LTS as your WSL2 distro

-ROS2 Jazzy installed and sourced (https://docs.ros.org/en/jazzy/Installation.html)

-Python 3.12 installed in WSL2

-VSCode with the Remote - WSL extension

## File Structure

-Inside the WSL terminal, make a ros2_ws directory

-Following steps on ROS2 download documentation, download ROS2 Jazzy

-Make a directory inside ros2_ws and call it src

## Setup Instructions
-Clone the repository

-Once cloned, you need to copy from Windows into WSL ROS2 Workspace

-This can be done in the WSL terminal

The general format:
```bash
cp -r /mnt/<drive-letter>/<path-to-your-repo>/* ~/ros2_ws/src/
cp -r /mnt/<drive-letter>/<path-to-your-repo>/.* ~/ros2_ws/src/ 2>/dev/null  # Include hidden files like .gitignore

Example for D: drive:
cp -r /mnt/d/csc3002-p158-giving-baxter-a-personality/* ~/ros2_ws/src/
```

Once this is copied, go to the root of the workspace (ros2_ws) and run
```bash
colcon build
```
Expect to see a build summary saying "Summary: X packages finished". Once this appears, run
```bash
source install/setup.bash
```
The general file structure format is:

```plaintext
ros2_ws/
└── src/
    ├── baxter_pkg/
    ├── requirements.txt
    └── setup.py
```

## Create and activate a virtual environment (Python 3.12)
### Before creating the virtual environment, make sure **venv support** is installed for Python 3.12:

```bash
sudo apt install python3.12-venv python3.12-dev -y
```

-Create virtual environment using Python 3.12
```bash
python3.12 -m venv venv
```
-Activate the virtual environment
```bash
source venv/bin/activate
```
-Check Python version
```bash
python3.12 --version  
```
-Upgrade pip (inside the venv)
```bash
python -m pip install --upgrade pip
```
-Install dependencies
```bash
pip install -r requirements.txt
```
-Install APT dependencies
```bash
sudo apt install portaudio19-dev python3.12-dev espeak-ng pulseaudio -y
```
## Pre PulseAudio sync

-edit ~/.bashrc in WSL terminal by using:
```bash
sudo nano ~/.bashrc
```
-Add the following line at the end: 
```bash
export PULSE_SERVER=tcp:$(/sbin/ip route | awk '/default/ { print $3 }')
```
-Finally source the bashrc
```bash
source ~/.bashrc
```
## PulseAudio

-Download PulseAudio installer from this link: https://pgaskin.net/pulseaudio-win32/

-Once installed, head to where PulseAudio is installed (Program Files x86) and go to etc/pulse

-Edit the 'default.pa' PA file in notepad, find where the following first line begins and paste it in

```bash
### Automatically load driver modules depending on the hardware available
.ifexists module-detect.so
### Use the static hardware detection module (for systems that lack udev support)
load-module module-detect
.endif

### Automatically connect sink and source if JACK server is present
.ifexists module-jackdbus-detect.so
.nofail
load-module module-jackdbus-detect channels=2
.fail
.endif


### Load several protocols
.ifexists module-esound-protocol-unix.so
load-module module-esound-protocol-unix
.endif
load-module module-native-protocol-unix

### Network access (may be configured with paprefs, so leave this commented
### here if you plan to use paprefs)
#load-module module-esound-protocol-tcp
#load-module module-native-protocol-tcp


### Make some devices default
#set-default-sink output
#set-default-source input

### Allow including a default.pa.d directory, which if present, can be used
### for additional configuration snippets.
.nofail
.include /pulseaudio/etc/pulse/default.pa.d
```


-Once changed, run command prompt as admin

-CD into where PulseAudio is installed e.g. 
```bash
cd C:\Program Files (x86)\PulseAudio\bin
```
-Once inside, run 
```bash
pulseaudio.exe --system --disallow-module-loading=0 --verbose
```

## PulseAudio notes
-A common issue depending on machine resources is PCM underruns. These occur when audio chunks are being requested before they are sent

-To try and play around with different buffer sizes, you can change the CHUNK_SIZE global variable to a different value

-Also inside of the 'daemon.conf' in your PulseAudio directory, at the bottom of the file you can experiment with different 'default-fragments' and 'default-fragments-size-msec' values to mitigate these underruns.

## Running the Chatbot
Make sure you are in the baxter_pkg package, have ROS2 sourced and dependencies installed

Do this inside of the venv, then run the chatbot using:
```bash
python3 src/baxter_pkg/chatbot_node.py
```

## Interacting with the Chatbot
Once the chatbot program has finished playing its greeting, a message in the terminal will appear saying "Listening for user input..."

Simply speak into your microphone, and Baxter will respond.

To change its personality, say "Change personality to X" 

X can be evil, funny, professional or else friendly.

## Further developing the Chatbot
If you wish to change anything in the program code, do so and build the program using:
```bash
colcon build --packages-select baxter_pkg
```

Then just run the program and see your changes in effect
## Troubleshooting

-If microphone input doesn't work inside WSL2, check that PulseAudio is running properly on Windows.

-For PCM underrun issues, tweak CHUNK_SIZE in the chatbot code or adjust buffer sizes in 'daemon.conf'.

-If a package fails during `pip install`, ensure your virtual environment is active and Python version is 3.12.

-Considering that the project was developed with Python 3.12, there will most likely be library issues if you want to upgrade the version of python

-If you are looking to add more libraries to the project, ensure that they are compatible with both Python 3.12 and Ubuntu 24.04
