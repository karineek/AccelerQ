# Troubleshooting Docker
sudo apt-get install docker.io
docker build -t dockerfile .

# If permission is denied, try to add a Docker group:
sudo groupadd docker
sudo usermod -aG docker $USER

# If still not working, log out and in: 
su - $USER
