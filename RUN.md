Boot the system.
Make the head straight, pointing towards the camera/under the camera.


## Control server (sends GCODE to the machine)
Open a terminal window
Login to the raspberry
$ ssh pi@192.168.0.29
$ tmux
$ cd riverbed-vision/control
$ ./control_server.py


## Brain
Open another terminal window
Login to the raspberry
$ ssh pi@192.168.0.29
$ tmux
$ cd riverbed-vision/brain
$ python brain.py



## Killing/Resume
If the emergency stop is pressed, the Brain will crash. The control server will still be running.
Make sure the Emergency button is not pressed anymore, otherwise homing will fail horribly.

So to resume, log in to the raspberry pi
$ ssh pi@192.168.0.29
$ tmux list-sessions

Pick the session for the brain
$ tmux attach -t 1

Restart the brain
$ python brain.py
