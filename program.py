import sympy as sp
import numpy as np
import logging
import threading
import re
import argparse
import json

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def result(self):
        return self._return

def array2str(array):
    result = re.sub('\s+', ' ', np.array_str(np.array(array).reshape(-1)))
    result = result.replace('[ ', '[').replace(' ]', ']')

    return result

def get_logger():
    logger = logging.getLogger('solve')
    logger.setLevel(logging.INFO)

    ch1 = logging.FileHandler('./logging.txt')
    ch2 = logging.StreamHandler()
    ch2.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s]%(message)s', '%H:%M:%S')

    ch1.setFormatter(formatter)
    ch2.setFormatter(formatter)

    logger.addHandler(ch1)
    logger.addHandler(ch2)

    return logger

def get_parameters_dict(list_of_parameters):
    if not len(list_of_parameters) == 8: raise Exception('wrong length of parameters list!')

    parameters = dict()
    parameters['k_x'], parameters['q'], parameters['n'], parameters['eps'], parameters['delta'], parameters['variables'], parameters['logger'], parameters['uw_initial'] = list_of_parameters

    return parameters

def set_parameters_dict_item(parameters, key_value_tuple):
    key, value = key_value_tuple
    parameters[key] = value

    return parameters

def get_variables(n):
    u_negative = sp.Symbol('u(-1)')
    w_negative = sp.Symbol('u(-1)')
    u = sp.symbols(f'u({0}:{n+2})')
    w = sp.symbols(f'w({0}:{n+2})')

    return u + (u_negative,), w + (w_negative,)

def first_equation(parameters, i):
    k_x, n, delta = parameters['k_x'], parameters['n'], parameters['delta']
    u, w = parameters['variables']

    equation = (u[i+1]-2*u[i]+u[i-1])/(delta**2)-k_x*(w[i+1]-w[i-1])/(2*delta)+(w[i+1]-w[i-1])/(2*delta)*(w[i+1]-2*w[i]+w[i-1])/(delta**2)
    
    boundary_conds = [(u[0], 0), (u[n], 0), 
                      (w[0], 0), (w[n], 0), 
                    # (u[-1], -u[1]), (u[n+1], -u[n-1]), 
                      (w[-1], -w[1]), (w[n+1], -w[n-1])]

    equation = equation.subs(boundary_conds)

    return equation

def second_equation(parameters, i):
    k_x, q, n, delta = parameters['k_x'], parameters['q'], parameters['n'], parameters['delta']
    u, w = parameters['variables']

    equation = -1/12*(w[i+2]-4*w[i+1]+6*w[i]-4*w[i-1]+w[i-2])/(delta**4)+k_x*((u[i+1]-u[i-1])/(2*delta)-k_x*w[i]+1/2*((w[i+1]-w[i-1])/(2*delta))**2)+(w[i+1]-w[i-1])/(2*delta)*((u[i+1]-2*u[i]+u[i-1])/(delta**2)-k_x*(w[i+1]-w[i-1])/(2*delta)+(w[i+1]-w[i-1])/(2*delta)*(w[i+1]-2*w[i]+w[i-1])/(delta**2))+(w[i+1]-2*w[i]+w[i-1])/(delta**2)*((u[i+1]-u[i-1])/(2*delta)-k_x*w[i]+1/2*((w[i+1]-w[i-1])/(2*delta))**2)+q
    
    boundary_conds = [(u[0], 0), (u[n], 0), 
                      (w[0], 0), (w[n], 0), 
                    # (u[-1], -u[1]), (u[n+1], -u[n-1]), 
                      (w[-1], -w[1]), (w[n+1], -w[n-1])]

    equation = equation.subs(boundary_conds)

    return equation

def get_equations(parameters):
    n = parameters['n']

    equations = list()

    for i in range(1, n):
        equation = first_equation(parameters, i)
        equations.append(equation)

    for i in range(1, n):
        equation = second_equation(parameters, i)
        equations.append(equation)

    return equations

def solve(parameters):
    tn = threading.current_thread().name
    k_x, q, n, eps, logger, uw_initial = parameters['k_x'], parameters['q'], parameters['n'], parameters['eps'], parameters['logger'], parameters['uw_initial']
    u, w = parameters['variables']

    logger.info(f'\t [{tn}]\t solve({k_x}, {q}, {n}, {eps})')

    uw = u[1:n] + w[1:n]
    equations = get_equations(parameters)

    uw_matrix = sp.Matrix(uw)
    equations_matrix = sp.Matrix(equations)
    W = equations_matrix.jacobian(uw_matrix)
    counter = 0

    logger.info(f'\t [{tn}]\t iteration #{counter}')

    x0 = uw_initial
    uw_tuples = list(zip(uw, x0.reshape(-1)))

    W0 = sp.matrix2numpy(W.subs(uw_tuples), dtype=np.float64)
    f0 = sp.matrix2numpy(equations_matrix.subs(uw_tuples), dtype=np.float64)
    x1 = x0 - np.dot(np.linalg.inv(W0), f0)
    counter += 1

    while np.max(np.abs(x1 - x0)) > eps:
        logger.info(f'\t [{tn}]\t iteration #{counter}')
        
        x0 = x1
        uw_tuples = list(zip(uw, x0.reshape(-1)))
        W0 = sp.matrix2numpy(W.subs(uw_tuples), dtype=np.float64)
        f0 = sp.matrix2numpy(equations_matrix.subs(uw_tuples), dtype=np.float64)
        x1 = x0 - np.dot(np.linalg.inv(W0), f0)
        counter += 1

    logger.info(f'\t [{tn}]\t [uw]\t {array2str(x1)}')

    return x1

def create_args_dicts(parameters):
    dicts = list()

    dicts_count = len(parameters['q'])

    for i in range(dicts_count):
        dictionary = get_parameters_dict([parameters['k_x'], parameters['q'][i], parameters['n'], parameters['eps'], parameters['delta'], parameters['variables'], parameters['logger'], parameters['uw_initial']])

        dicts.append(dictionary)
    
    return dicts

def create_threads(parameters):
    threads_count = len(parameters['q'])
    args = create_args_dicts(parameters)
    threads = list()

    for i in range(threads_count):
        threads.append(ThreadWithReturnValue(target=solve, args=[args[i]], name=f'{i}'))

    return threads

def get_cmd_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-lbi', type=int, help='left_bound_index', default=0)
    parser.add_argument('-rbi', type=int, help='right_bound_index', default=25)
    parser.add_argument('-tc', type=int, help='thread_count', default=4)
    parser.add_argument('-k', type=float, help='k_x', default=24.0)
    parser.add_argument('-n', type=int, help='n', default=10)
    parser.add_argument('-e', type=float, help='eps', default=1e-3)
    parser.add_argument('-uw', type=json.loads, help='uw_initial', default=[])

    args = parser.parse_args()

    if not args.rbi > 0:
        raise Exception('rbi >= 0')
    
    if not args.lbi >= 0:
        raise Exception('lbi > 0')

    if not args.lbi < args.rbi:
        raise Exception('wrong lb and rbi arguments!')

    if not args.tc > 1:
        raise Exception('wrong tc argument!')

    if not args.n >= 0:
        raise Exception('n >= 0')

    if len(args.uw) == 0 and args.n != 0 or len(args.uw) == 0 and args.n == 0:
        args.uw = [0] * (2*(args.n-1))

    if not len(args.uw) == 2*(args.n-1): 
        raise Exception('wrong uw length!')

    args.uw = np.array(args.uw).reshape((2*(args.n-1), 1))

    return args

# linear extrapolation
# https://en.wikipedia.org/wiki/Extrapolation
def lixp(y1, y2):
    if not len(y1) == len(y2):
        raise Exception('y_pair elements should be the same length!')

    y = np.full(y1.shape, 0.0)

    cols = len(y1)

    for j in range(cols):
        y[j] = 2.0*y2[j] - y1[j]

    return y

def main():
    parameters = dict()

    args = get_cmd_args()

    thread_count = args.tc
    k_x = args.k
    n = args.n
    eps = args.e
    uw_initial = np.array(args.uw)
    lbi, rbi = args.lbi, args.rbi

    delta = 1.0 / n
    variables = get_variables(n)
    logger = get_logger()

    with open('./logging.txt', 'w') as f:
        f.truncate()

    for j in range(lbi, rbi):
        
        step = float(thread_count)
        left_bound, right_bound = step*j, step*(j + 1)
        
        q = np.linspace(left_bound, right_bound, thread_count)
        parameters = get_parameters_dict([k_x, q, n, eps, delta, variables, logger, uw_initial])

        threads = create_threads(parameters)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        
        result = list()

        for thread in threads:
            result.append(thread.result())

        w = list()

        for i in range(len(threads)):
            solution = result[i]
            w_len = int(len(solution) / 2)
            w_inc = solution.reshape(-1)[w_len:]
            w_max = np.max(np.concatenate([[0], w_inc, [0]]))
            w.append(w_max)

        logger.info(f'\t [q]\t {array2str(q)}')
        logger.info(f'\t [w]\t {array2str(w)}\n')

        uw_initial = lixp(result[-2], result[-1])

if __name__ == '__main__':
    main()