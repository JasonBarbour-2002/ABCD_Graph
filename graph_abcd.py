'''
Description: This is a wrapper class for the ABCD graph generator written in Julia.
'''
import os
import subprocess
import tempfile
import toml
import networkx as nx
import numpy as np


class GraphABCD:
    '''
    This is a Wrapper class for the ABCD graph generator written in Julia.
    It is used to generate a graph with the given parameters. 
    The link to the original repository is https://github.com/bkamins/ABCDGraphGenerator.jl
    '''
    def __init__(self,n:int,t1:float,d_min:int,
                d_max:int,t2:float,
                c_min:int,c_max:int,c_max_iter:int=1000,d_max_iter:int=1000,
                xi:float=None,mu:float=None,
                islocal:bool=False,is_cl:bool=False,nout:int=0,
                seed:int=None,path:str = None):
        '''
        This is a wrapper class for the ABCD graph generator written in Julia.
        It is used to generate a graph with the given parameters. 
        The link to the original repository is https://github.com/bkamins/ABCDGraphGenerator.jl
        
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
        d_max_iter, optional
            maximum number of iterations for sampling degrees. Default is 1000
        t2
            power-law exponent for cluster size distribution    
        c_min
            minimum community size
        c_max
            maximum number of communities
        c_max_iter, optional
            maximum number of iterations for sampling cluster sizes. Default is 1000
        xi
            fraction of edges to fall in background graph
        mu
            mixing parameter
        !Exactly one of xi and mu must be passed as Float.
        Also if xi is provided islocal must be set to false or omitted.
        
        islocal, optional
            if "true" mixing parameter is restricted to local cluster, otherwise it is global
        
        is_cl, optional
            if "false" use configuration model, if "true" use Chung-Lu
            isCL = "false", and xi (not mu) must be passed
        nout, optional
            number of vertices in graph that are outliers; optional parameter
        if nout is passed and is not zero then we require islocal = "false",
        if nout > 0 then it is recommended that xi > 0
        seed, optional
            seed for the random number generator
        path, optional
            the path to the directory where you want to save the generated network info.
            If an empty string is passed, the current directory is used,
            otherwise the information is saved in the given directory.
            If omitted, the information is not saved.
        '''
        self.n = n
        self.t1 = t1
        self.d_min = d_min
        self.d_max = d_max
        self.d_max_iter = d_max_iter
        self.t2 = t2
        self.c_min = c_min
        self.c_max = c_max
        self.c_max_iter = c_max_iter
        if xi is None and mu is None:
            raise ValueError("xi and mu cannot be None at the same time")
        if xi is not None and mu is not None:
            raise ValueError("Either xi or mu must be None")
        self.xi = xi
        self.mu = mu
        self.islocal = 'true' if islocal else 'false'
        self.is_cl = 'true' if is_cl else 'false'
        self.nout = nout
        self.seed = '' if seed is None else seed
        self.tempfile = False
        self.__path__maker(path)
        self.__setup()

    def modify_args(self,**kwargs):
        """
        Modifies the parameters of the graph generator.
        """
        for key,val in kwargs.items():
            if key == 'mu':
                self.mu = val
                self.xi = None
                if 'xi' in self.args:
                    del self.args['xi']
                self.args['mu'] = str(self.mu)
            elif key == 'xi':
                self.xi = val
                self.mu = None
                if 'mu' in self.args:
                    del self.args['mu']
                self.args['xi'] = str(self.xi)
            elif key == 'path':
                self.__path__maker(val)
            elif isinstance(val, bool):
                self.args[key] = 'true' if val else 'false'
            else:
                self.args[key] = str(val)

    def __path__maker(self,path):
        if path is None:
            self.tempfile = True
            self.path = ''
        elif path == '':
            self.path = path
        else:
            self.path = path+('/'if path[-1] != '/'else '')
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        self.path_dir = None

    def generate(self):
        '''
        Generates a graph with the given parameters.
        
        Returns
        -------
        A networkx graph object with the community attribute set for each node.
        '''
        args = self.args.copy()
        path = self.path
        if self.tempfile:
            temp = tempfile.TemporaryDirectory()
            path = temp.name
        args = self.__add_file_paths(path,args)
        path_toml = 'ABCDGraphGenerator/utils/Graph.toml'
        with open(path_toml,'w',encoding='utf-8') as file_name:
            toml.dump(args,file_name)
        julia_file_path = 'ABCDGraphGenerator/utils/abcd_sampler.jl'
        try:
            self.__call_julia_file(julia_file_path, [path_toml])
        except subprocess.CalledProcessError as exc:
            raise ValueError('The graph could not be generated. '
            'Try to generate the graph again. If the problem persists, '
            'try to choose different parameters.') from exc
        net = nx.read_edgelist(path+"edge.dat", nodetype=int)
        community = np.loadtxt(path+"com.dat", dtype=int)
        sort  = np.argsort(community[:,0])
        community = community[sort]
        community = {i+1:community[i,1] for i in range(len(community))}
        nx.set_node_attributes(net,community,'community')
        if self.tempfile:
            temp.cleanup()
        return net

    def __call_julia_file(self,julia_file_path, args):
        # Build the command to execute the Julia file
        julia_executable = subprocess.check_output(["which", "julia"]).strip()
        command = [julia_executable, "--startup-file=no", julia_file_path] + args

        # Execute the command and capture the output
        result = subprocess.check_output(command, universal_newlines=True)

        # Return the result
        return result

    def __setup(self):
        args = {}
        args['n'] = str(self.n)
        args['t1'] = str(self.t1)
        args['d_min'] = str(self.d_min)
        args['d_max'] = str(self.d_max)
        args['d_max_iter'] = str(self.d_max_iter)
        args['t2'] = str(self.t2)
        args['c_min'] = str(self.c_min)
        args['c_max'] = str(self.c_max)
        args['c_max_iter'] = str(self.c_max_iter)
        if self.xi is not None and self.mu is None:
            args['xi'] = str(self.xi)
        elif self.xi is None and self.mu is not None:
            args['mu'] = str(self.mu)
        else :
            raise ValueError("Only one of xi and mu must be passed")
        args['islocal'] = self.islocal
        args['isCL'] = self.is_cl
        args['nout'] = str(self.nout)
        args['seed'] = ''if self.seed is None else self.seed
        self.args = args

    def __add_file_paths(self,path,args):
        args = args.copy()
        args['degreefile'] = path+'deg.dat'
        args['communitysizesfile'] = path+'cs.dat'
        args['communityfile'] = path+'com.dat'
        args['networkfile'] = path+'edge.dat'
        return args
