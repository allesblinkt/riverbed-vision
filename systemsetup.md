
## archlinux Arm installation

Follow instructions given on

https://archlinuxarm.org/platforms/armv8/broadcom/raspberry-pi-3



## First login

```sh
$ ssh alarm@alarmpi
PW: alarm
```


## User

### Install sudo

```sh 
$ su
PW: root
$ pacman -Syu
$ pacman -S sudo
```

Enable sudoing via group wheel

```sh
$ visudo
```

Uncomment 

```
%wheel ALL=(ALL) ALL
```


### Make riverbed user

```sh
$ useradd -m -G wheel -s /bin/bash riverbed
$ passwd riverbed
```

TODO: groups

### Delete default user

```sh
$ sudo userdel alarm
$ sudo rm -rf /home/alarm
```




## General setup

### Hostname

```sh
$ sudo hostnamectl set-hostname riverbed
```

### Login message

edit `/etc/motd`


### Avahi / Zeroconf

```sh
$ sudo pacman -S avahi nss-mdns
$ sudo systemctl enable avahi-daemon.service
$ sudo systemctl start avahi-daemon.service

```

#### /etc/nsswitch.conf 

change the line:

```
hosts: files dns myhostname
```

```
hosts: files mdns_minimal [NOTFOUND=return] dns myhostname
```




## Packages

### Convenience packages

```sh
$ sudo pacman -S htop vim git
$ sudo pacman -S binutils base-devel python ipython
```


### Development

$ sudo pacman -S opencv	python-numpy

