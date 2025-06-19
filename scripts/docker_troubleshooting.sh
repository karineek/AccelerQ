Troubleshooting Docker
sudo apt-get install docker.io
docker build -t dockerfile .


if permission denied try to add a docker group:
sudo groupadd docker
sudo usermod -aG docker <username>

if still not working log out and in: 
su - $USER
