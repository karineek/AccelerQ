# Troubleshooting Docker
```
sudo apt-get install docker.io
docker build -t dockerfile .
```

## If permission is denied, try to add a Docker group:
```
sudo groupadd docker
sudo usermod -aG docker $USER
```

## If still not working, log out and in: 
```
su - $USER
```

# Further Help:
Before that, make sure you have Docker installed with the right permissions:
```
## Remove all old versions of Docker
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

## Install Docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo groupadd docker
## Fix permissions
sudo gpasswd -a $USER docker
sudo usermod -aG docker $USER

## Restart the service
sudo systemctl restart docker

## Test if it is all okay
docker --version
docker run hello-world
```
See further help [here](https://docs.docker.com/engine/install/ubuntu/).

To remove some old docker image and containers, you first need to see which one are still in your system (usually as hidden files):
```
docker system df -v
```
You then can remove images with ```docker rmi <hash-image>``` and containers with ```docker rm <hash-container>```.
