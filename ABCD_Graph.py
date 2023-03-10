import subprocess
import toml
import networkx as nx
import os
import numpy as np

def call_julia_file(julia_file_path, args):
    # Build the command to execute the Julia file
    julia_executable = subprocess.check_output(["which", "julia"]).strip()
    command = [julia_executable, "--startup-file=no", julia_file_path] + args

    # Execute the command and capture the output
    result = subprocess.check_output(command, universal_newlines=True)

    # Return the result
    return result

def ABCD_Graph(n:int,t1:float,d_min:int,d_max:int,d_max_iter:int,t2:float,c_min:int,c_max:int,c_max_iter:int,xi:float,mu:float=None,islocal:bool=False,isCL:bool=False,degreefile:str='deg.dat',communitysizesfile:str='cs.dat',communityfile:str='com.dat',networkfile:str='edge.dat',nout:int=0,seed:int=None,path:str = None)-> nx.Graph:
    '''> This function generates a network with the given parameters
    
    Parameters
    ----------
    n
        number of nodes
    t1
        power-law exponent for degree distribution
    d_min
        minimum degree
    d_max
        maximum degree
    d_max_iter
        maximum number of iterations for sampling degrees
    t2
        power-law exponent for cluster size distribution
    c_min
        minimum community size
    c_max
        maximum number of communities
    c_max_iter
        maximum number of iterations for sampling cluster sizes
    xi
        the parameter of the power law distribution of the community sizes
    mu
        the average degree of the network
    ! Exactly one of xi and mu must be passed as Float64. Also if xi is provided islocal must be set to false or omitted.

    islocal, optional
        if "true" mixing parameter is restricted to local cluster, otherwise it is global
    generate a global community structure.
    isCL, optional
        if "false" use configuration model, if "true" use Chung-Lu
        isCL = "false", and xi (not mu) must be passed
    degreefile, optional
        the file where the degree sequence is stored
    communitysizesfile, optional
        The file where the community sizes will be written.
    communityfile, optional
        the file where the community labels will be written
    networkfile, optional
        the name of the file where the network will be saved.
    nout, optional
        number of vertices in graph that are outliers; optional parameter
        if nout is passed and is not zero then we require islocal = "false",
        if nout > 0 then it is recommended that xi > 0
    seed, optional
        seed for the random number generator
    path, optional
        the path to the directory where you want to save the generated network info.
    
    Returns
    -------
        A networkx graph object with the community attribute set for each node.
    
    '''
    
    data = locals()
    path = (path+('/'if path[-1] != '/'else '')) if path != None else ''

    if path != '':
        if not os.path.exists(path):
            os.makedirs(path)
    del data['path']
    if seed == None:
        data['seed'] = ''
    if nout == 0:
        del data['nout']
    if xi == None and mu == None:
        raise ValueError("xi and mu cannot be None at the same time")
    if xi != None and mu != None:
        raise ValueError("Either xi or mu must be None")
    if mu == None:
        del data['mu']
    if xi == None:
        del data['xi']
    for key in list(data.keys()):
        if data[key] == False or data[key] == True:
            if bool(data[key]):
                data[key] = 'true'
            else:
                data[key] = 'false'
        if data[key] is not str:
            data[key]= str(data[key])
        if 'file' in key:
            data[key] = path+ data[key]
    F = open("ABCDGraphGenerator/utils/Graph.toml", "w")
    data = toml.dump(data,F)
    F.close()
    call_julia_file("ABCDGraphGenerator/utils/abcd_sampler.jl",["ABCDGraphGenerator/utils/Graph.toml"])
    net = nx.read_edgelist(path+"edge.dat", nodetype=int)
    community = np.loadtxt(path+"com.dat", dtype=int)
    sort  = np.argsort(community[:,0])
    community = community[sort]
    community = {i+1:community[i,1] for i in range(len(community))}
    nx.set_node_attributes(net,community,'community')
    return net
