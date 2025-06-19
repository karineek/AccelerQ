homedir=$1 # e.g. /home/ubuntu

sudo fallocate -l 2G $homedir/swapfile
sudo chmod 600 $homedir/swapfile
sudo mkswap $homedir/swapfile
sudo swapon $homedir/swapfile

sudo fallocate -l 4G $homedir/swapfile-4G
sudo chmod 600 $homedir/swapfile-4G
sudo mkswap $homedir/swapfile-4G
sudo swapon $homedir/swapfile-4G

swapon --show
sudo sysctl vm.vfs_cache_pressure=200 # to avoid OOM killed issues
