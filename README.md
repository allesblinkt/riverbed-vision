# riverbed-vision

## Installation on Raspberry Pi

Install packages:

```
sudo apt-get update
sudo apt-get install python-netifaces python-opencv python-serial python-skimage
sudo pip install Pyro4
```

Clone the repository:

```
git clone https://github.com/allesblinkt/riverbed-vision.git
```

## Others

To turn off the serial console:

```
sudo systemctl mask serial-getty@ttyAMA0.service
```

and remove `console=ttyAMA0,115200` from `/boot/cmdline.txt`.

Then reboot.
