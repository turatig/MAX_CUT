import subprocess
import sys

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Args file name required")
        exit()
    with open(sys.argv[1]) as f:
        for line in f:
            cmd=["../main"]
            line=line.split(" ")
            cmd.append(line[1])
            cmd.append(line[3])
            cmd.append(line[5][:-1])
            subprocess.run(cmd)
        f.close()

