# 1) Generate the virtual environment:

python3 -m venv .
# (Probably you'll have to install venv)

# 2) source the virtual environment, you should see .veng before user@PC:~/ros_ws
source .venv/bin/activate 
# the source can be added to the .bashrc file
# the .bashrc has to be like this:
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
source ~/ros_ws/.venv/bin/activate
export ROS_DOMAIN_ID=10
export ROS_LOCALHOST_ONLY=1
source ~/ros_ws/install/setup.bash

# 2) Install the requirements (be sure to be in virtual environment):

pip install -r requirements.txt



