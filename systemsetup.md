## Disable bluetooth and use UART normally

/boot/config.txt

```
...
dtoverlay=pi3-disable-bt
...
```

TODO: kernel console
systemd




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
$ sudo usermod -a -G uucp,video riverbed
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









## Access Point, DHCP, NAT

HostAPD seems to work fine out of the box (no special drivers) with a **TP TL-WN722N** USB Dongle.


```sh
$ pacman -S hostapd
```

This is partly based on this WIKI page: https://wiki.archlinux.org/index.php/Software_access_point


### /etc/hostapd/hostapd.conf

```
interface=wlan0
driver=nl80211
ssid=River_📠
hw_mode=g
channel=6
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=riverofjoy
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
```

### Test 

```
$ sudo ip address change 10.0.42.2/24 dev wlan0
$ sudo hostapd /etc/hostapd/hostapd.conf
```

### Service

```
$ sudo systemctl enable hostapd.service
$ sudo systemctl start hostapd.service
```

### Set static IP for wlan0 at boot

/etc/systemd/network/wlan0_ap_static.network 

```
[Match]
Name=wlan0

[Network]
Address=10.0.42.42/24
```

```sh
$ sudo systemctl restart systemd-networkd.service.
```


networkctl list


### Forwarding enable


```sh
$ sudo sysctl net.ipv4.ip_forward=1
```

Make it persistent...


#### /etc/sysctl.d/30-ipforward.conf

```
net.ipv4.ip_forward=1
net.ipv6.conf.default.forwarding=1
net.ipv6.conf.all.forwarding=1
```


### NAT

```sh
$ sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
$ sudo iptables -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
$ sudo iptables -A FORWARD -i wlan0 -o eth0 -j ACCEPT
```

and save them

```sh
$ sudo su
$ iptables-save > /etc/iptables/iptables.rules
$ sudo systemctl enable iptables.service
$ sudo systemctl start iptables.service
```

You can check after a reboot, if they got applied:

```sh
$ sudo iptables --list-rules
```

TODO: firewall



### DHCP, DNS via dnsmasq


#### /etc/dnsmasq.conf

```
port=53
interface=wlan0
bind-interfaces
dhcp-range=10.0.42.50,10.0.42.150,12h
```


```sh
$ sudo systemctl enable dnsmasq.service
$ sudo systemctl start dnsmasq.service
```
