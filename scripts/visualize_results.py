import matplotlib.pyplot as plt

if __name__=="__main__":
    log_files=["hopfield_sequential.txt","hopfield_parallel.txt",\
                "lorena_sequential.txt","lorena_parallel.txt"]
    
    size=[]
    hs_time=[]
    hs_cost=[]

    with open("hopfield_sequential.txt") as f:
        for line in f:
            line=line.split(" ")
            if line[0]=="Size:":
                size.append(int(line[1][:-1]))
            else:
                hs_time.append(float(line[1]))
                hs_cost.append(int(line[3][:-1]))
        f.close()

    hp_time=[]

    with open("hopfield_parallel.txt") as f:
        for line in f:
            line=line.split(" ")
            if line[0]!="Size:":
                hp_time.append(float(line[1]))
        f.close()

    fig,ax=plt.subplots()
    ax.set_title("Hopfield network-computation times")
    ax.set_xlabel("Size (number of nodes)")
    ax.set_ylabel("Time")
    ax.plot(size,hs_time,label="Sequential implementation",color="blue")
    ax.plot(size,hp_time,label="Parallel implementation",color="red")
    ax.legend()
    fig.savefig("Hopfield_time.png")
    
    ls_map_time=[]
    ls_cut_time=[]
    ls_cost=[]

    with open("lorena_sequential.txt") as f:
        for line in f:
            line=line.split(" ")
            if line[0]!="Size:":
                if line[0]=="(Map)":
                    ls_map_time.append(float(line[2]))
                else:
                    ls_cut_time.append(float(line[2]))
                    ls_cost.append(int(line[4]))
        f.close()
    
    
    lp_map_time=[]
    lp_cut_time=[]
    with open("lorena_parallel.txt") as f:
        for line in f:
            line=line.split(" ")
            #print(line)
            if line[0]!="Size:":
                if line[0]=="(Map)":
                    lp_map_time.append(float(line[2]))
                else:
                    lp_cut_time.append(float(line[2]))
        f.close()

    fig,ax=plt.subplots()
    ax.set_title("Lorena(Map on unit circle)-computation times")
    ax.set_xlabel("Size (number of nodes)")
    ax.set_ylabel("Time")
    ax.plot(size,ls_map_time,label="Sequential implementation",color="blue")
    ax.plot(size,lp_map_time,label="Parallel implementation",color="red")
    ax.legend()
    fig.savefig("Lorena_map_time.png")

    fig,ax=plt.subplots()
    ax.set_title("Lorena(Find maximum cut)-computation times")
    ax.set_xlabel("Size (number of nodes)")
    ax.set_ylabel("Time")
    ax.plot(size,ls_cut_time,label="Sequential implementation",color="blue")
    ax.plot(size,lp_cut_time,label="Parallel implementation",color="red")
    ax.legend()
    fig.savefig("Lorena_cut_time.png")

        
    fig,ax=plt.subplots()
    ax.set_title("Hopfield vs Lorena-solution's cost")
    ax.set_xlabel("Size (number of nodes)")
    ax.set_ylabel("Time")
    width=5
    ax.bar(size,hs_cost,width,label="Hopfield network",color="blue")
    ax.bar([i+width for i in size],ls_cost,width,label="Lorena",color="red")
    ax.legend()
    fig.savefig("Cost.png")

    #print("-"*20+"Data visualized"+"-"*20)
