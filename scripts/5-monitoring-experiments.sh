cnum=$1

echo "Docker ==>"
docker exec -it "$cnum" cat /home/kclq/AccelerQ/src/logger-itr.txt
echo ">> Statistics:"
free -h ; uptime

sudo sync
sudo sysctl -w vm.drop_caches=3
