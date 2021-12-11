import numpy as np

class VecWrap:

    def __init__(self, params):
        self.params = params

    def print(self):
        print(self.params)

    def __str__(self):
        s='['
        i=0
        for key, p in self.params.items():
            if p.ndim == 2:
                s+='%s:%dx%d' % (key,p.shape[0], p.shape[1])
            else:
                s+='%s:%d' % (key,p.shape[0])
            if i != len(self.params)-1:
                s+=', '
            i += 1
        s+=']'
        return s

    def __add__(self, other):
        assert self.params.keys() == other.params.keys(), print(self.params.keys() , other.params.keys())
        return VecWrap({ key : self.params[key] + other.params[key] for key in self.params.keys()})
        
    def __sub__(self, other):
        assert self.params.keys() == other.params.keys(), print(self.params.keys() , other.params.keys())
        return VecWrap({ key : self.params[key] - other.params[key] for key in self.params.keys()})

    def __mul__(self, scalar):
        return VecWrap({ key : scalar * val for key, val in self.params.items()})

    def __iadd__(self, other):
        assert self.params.keys() == other.params.keys(), print(self.params.keys() , other.params.keys())
        for key in self.params.keys():
            self.params[key] += other.params[key]
        return VecWrap(self.params)

    def __isub__(self, other):
        assert self.params.keys() == other.params.keys(), print(self.params.keys() , other.params.keys())
        for key in self.params.keys():
            self.params[key] -= other.params[key]
        return VecWrap(self.params)

    def __imul__(self, scalar):
        for key in self.params.keys():
            self.params[key] *= scalar
        return VecWrap(self.params)

    def norm_sq(self):
        val = 0
        for w in self.params.values():
            val += np.linalg.norm(w)**2
        return val

    def dot(self, other):
        assert self.params.keys() == other.params.keys(), print(self.params.keys() , other.params.keys())
        v = 0
        for key in self.params.keys():
            v += np.dot(self.params[key].ravel(), other.params[key].ravel())
        return v

    def __neg__(self):
        return __imul__(self,-1)

    __rmul__ = __mul__

class DictToArray:

    def __init__(self,params):
        self.shapes={}
        self.locs={}
        ind=0
        for k,v in params.items():
            self.shapes[k]=v.shape
            ind1 = ind + v.size
            self.locs[k]=[ind,ind1]
            ind = ind1
        self.tot_size=ind

    def forward(self,params):
        x=np.zeros(self.tot_size)
        for k,v in params.items():
            x[self.locs[k][0]:self.locs[k][1]]=v.ravel()
        return x

    def backward(self,x):
        params={}
        for k, loc in self.locs.items():
            m=x[loc[0]:loc[1]]
            mm=np.reshape(m,self.shapes[k])
            params[k]=mm
        return params







        







    
