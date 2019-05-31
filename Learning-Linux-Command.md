#### counting the number of files in a directory

ls directory | wc -l

#### checking the size of the file

ls -sh

#### print the information of the system on spartan

motd

#### move all files in subfolders to its parent folder

find . -mindepth 2 -type f -print -exec mv {} . \;

##### from Downloads to Videos

find ~/Downloads/ -type f -print0 | xargs -0 mv -t ~/Videos