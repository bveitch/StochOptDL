import numpy as np
from vecWrap import DictToArray

def test(dict_in, test_name):

    converter=DictToArray(dict_in)
    array = converter.forward(dict_in)
    dict_out =converter.backward(array)

    assert dict_in.keys() == dict_out.keys(), print(dict_in.keys(), dict_out.keys())
    passed = True
    for k in dict_in.keys():
        passed *= (dict_in[k]==dict_out[k]).all()
    if passed:
        print('%s passed' % test_name) 
    return passed

if __name__ == '__main__':
    d1={'w':np.ones((5,3))}
    test(d1,'test1')
    d2={'w1': np.ones((7,3)), 'b1' : 2*np.ones(3)}
    test(d2,'test2')
    d3={'w1': np.random.normal(loc=0.0,scale=1.0,size=(1000,300)), 'b1' : np.zeros(300),'w2': np.random.normal(loc=0.0,scale=1.0,size=(300,100)), 'b2' : np.zeros(100)}
    test(d3,'test3')
    d4={'a': np.ones((1000,300)), 'b' : 0.5*np.ones(500),'c': np.zeros((300,100)), 'd' : 1.04*np.ones((600,100)), 'e': 5.1*np.ones(7)}
    test(d4,'test4')



