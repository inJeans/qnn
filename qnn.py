import platform
import os
import datetime
import logging
import copy

from scipy.linalg import expm
from pyquil.quil import Program
from pyquil.quilbase import DefGate
from pyquil.api import QVMConnection
from pyquil.gates import CNOT, X, TRUE, FALSE, NOT, RZ, RY, H
from pyquil.paulis import sY, exponential_map
from pyquil.parameters import Parameter, quil_sin, quil_cos

from tqdm import tqdm

import numpy as np

LOGGER = logging.getLogger("qnn")
DEBUG = False

NUM_INPUT = 2
NUM_HIDDEN = 2
NUM_OUTPUT = 1

ANCILLARY_BIT = NUM_INPUT + NUM_HIDDEN + 2*NUM_OUTPUT

NUM_SAMPLES = 10

def main():
    theta = 0.5 * np.pi
    # W1 = np.random.rand((NUM_INPUT, NUM_HIDDEN))
    # W2 = np.random.rand((NUM_HIDDEN, NUM_OUTPUT))
    W1 = np.ones(NUM_INPUT*NUM_HIDDEN) * theta
    b1 = theta
    W2 = np.ones(NUM_HIDDEN*NUM_OUTPUT) * theta
    b2 = theta
    weights = []
    weights.extend(W1)
    weights.append(b1)
    weights.extend(W2)
    weights.append(b2)
    weights = np.array(weights)

    error = error_estimate(weights)

    res = nelder_mead(error_estimate, weights, max_iter=3)

    print(res)

    print(forward_prop(res[0], initialization="test"))

def error_estimate(weights):

    output_sum = 0
    for _ in tqdm(range(NUM_SAMPLES)):
        output = forward_prop(weights)
        output_sum += output

    expectation_value = output_sum / NUM_SAMPLES

    return (0.5 - expectation_value)**2

def forward_prop(weights, initialization=None):
    LOGGER.info("Connecting to the QVM...")
    qvm = QVMConnection()
    LOGGER.info("... done")
    LOGGER.info(" ")

    LOGGER.info("Initialising quantum program...")
    p = Program()

    LOGGER.info("... defining custom gates")
    LOGGER.info("... controlled Ry")
    CRY = controlled_Ry(p)
    LOGGER.info("... controlled sY")
    CSY = controlled_sY(p)
    LOGGER.info("... done")
    LOGGER.info(" ")

    a = -1
    rotation = 0.5*np.pi*(a + 1)
    gamma = 1 / NUM_INPUT

    W1 = weights[:NUM_INPUT*NUM_HIDDEN].reshape((NUM_INPUT, NUM_HIDDEN))
    b1 = weights[NUM_INPUT*NUM_HIDDEN]
    W2 = weights[NUM_INPUT*NUM_HIDDEN+1:NUM_INPUT*NUM_HIDDEN+1+NUM_HIDDEN*NUM_OUTPUT].reshape(NUM_HIDDEN, NUM_OUTPUT)
    b2 = weights[-1]

    # INITIALISE INPUT
    if initialization == None:
        EDI = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        dg = DefGate("EDI", EDI)
        EDI = dg.get_constructor()

        p.inst(dg)
        p.inst(H(0))
        p.inst(H(1))
   
        p.inst(EDI(0, 1))
    if initialization == "test":
        # p.inst(X(1))  # |01>
        p.inst(X(0))  # |10>
        # pass

    # INTIALISE LABELS
    p.inst(H(NUM_INPUT + NUM_HIDDEN + NUM_OUTPUT))

    classical_flag_register = ANCILLARY_BIT + 1
    # Write out the loop initialization and body programs:
    for n in range(NUM_HIDDEN):
        loop_body = Program()
        for w in range(NUM_INPUT):
            loop_body.inst(CRY(4.*gamma*W1[w, n])(w, ANCILLARY_BIT))
        loop_body.inst(RY(2.*gamma*b1)(ANCILLARY_BIT))

        loop_body.inst(CSY(ANCILLARY_BIT, NUM_INPUT+n))
        loop_body.inst(RZ(-0.5*np.pi)(ANCILLARY_BIT))
        for w in range(NUM_INPUT):
            loop_body.inst(CRY(-4.*gamma*W1[w, n])(w, ANCILLARY_BIT))
        loop_body.inst(RY(-2.*gamma*b1)(ANCILLARY_BIT))

        loop_body.measure(ANCILLARY_BIT, classical_flag_register)
    
        then_branch = Program(RY(-0.5*np.pi)(NUM_INPUT+n))
        then_branch.inst(X(ANCILLARY_BIT))
        else_branch = Program()
 
        # Add the conditional branching:
        loop_body.if_then(classical_flag_register,
                          then_branch,
                          else_branch)

        init_register = Program(TRUE([classical_flag_register]))
        loop_prog = init_register.while_do(classical_flag_register,
                                           loop_body)
        p.inst(loop_prog)

    # Write out the loop initialization and body programs:
    for n in range(NUM_OUTPUT):
        loop_body = Program()
        for w in range(NUM_HIDDEN):
            loop_body.inst(CRY(4.*gamma*W2[w, n])(w, ANCILLARY_BIT))
        loop_body.inst(RY(2.*gamma*b2)(ANCILLARY_BIT))

        loop_body.inst(CSY(ANCILLARY_BIT, NUM_INPUT+NUM_HIDDEN+n))
        loop_body.inst(RZ(-0.5*np.pi)(ANCILLARY_BIT))
        for w in range(NUM_HIDDEN):
            loop_body.inst(CRY(-4.*gamma*W2[w, n])(w, ANCILLARY_BIT))
        loop_body.inst(RY(-2.*gamma*b1)(ANCILLARY_BIT))

        loop_body.measure(ANCILLARY_BIT, classical_flag_register)
    
        then_branch = Program(RY(-0.5*np.pi)(NUM_INPUT+NUM_HIDDEN+n))
        then_branch.inst(X(ANCILLARY_BIT))
        else_branch = Program()
 
        # Add the conditional branching:
        loop_body.if_then(classical_flag_register,
                          then_branch,
                          else_branch)

        init_register = Program(TRUE([classical_flag_register]))
        loop_prog = init_register.while_do(classical_flag_register,
                                           loop_body)
        p.inst(loop_prog)

    p.measure(NUM_INPUT+NUM_HIDDEN, 0)

    LOGGER.info("... executing on the QVM")
    classical_regs = [0]
    output = qvm.run(p, classical_regs)
    LOGGER.info("... %s", output)
    LOGGER.info("")

    return output[0][0]

def controlled(U):
    controlled_u = np.array([[ 1., 0., 0., 0.],
                             [ 0., 1., 0., 0.],
                             [ 0., 0., _ , _ ],
                             [ 0., 0., _ , _ ]])
    return controlled_u

def controlled_Ry(program):
    theta = Parameter('theta')

    cry = np.array([[ 1., 0., 0.         , 0.],
                    [ 0., 1., 0.         , 0.],
                    [ 0., 0., quil_cos(0.5*theta) , quil_sin(0.5*theta) ],
                    [ 0., 0.,-quil_sin(0.5*theta) , quil_cos(0.5*theta) ]])

    dg = DefGate('CRY', cry, [theta])
    program.inst(dg)
    
    return dg.get_constructor()

def controlled_sY(program):
    csy = np.array([[ 1., 0., 0.  , 0.  ],
                    [ 0., 1., 0.  , 0.  ],
                    [ 0., 0., 0.  ,-1.j ],
                    [ 0., 0., 1.j , 0.  ]])

    dg = DefGate('CSY', csy)
    program.inst(dg)
    
    return dg.get_constructor()

def Ry(t, qubit):
    return exponential_map(sY(qubit))(-0.5*t)

def nelder_mead(f, x_start,
               step=0.1, no_improve_thr=10e-6,
               no_improv_break=10, max_iter=0,
               alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres

def set_up_logger():
    """This function initialises the logger.

    We set up a logger that prints both to the console at the information level
    and to file at the debug level. It will store in the /temp directory on
    *NIX machines and in the local directory on windows.
    """
    timestamp = datetime.datetime.now()

    logfile_name = 'qnn-{0:04}-{1:02}-{2:02}-{3:02}{4:02}{5:02}.log'\
                   .format(timestamp.year,
                           timestamp.month,
                           timestamp.day,
                           timestamp.hour,
                           timestamp.minute,
                           timestamp.second)

    if platform.system() == 'Windows':
        logfile_name = './' + logfile_name
    else:
        logfile_name = '/tmp/' + logfile_name

    logging.basicConfig(filename=logfile_name,
                        level=logging.DEBUG)

    console_logger = logging.StreamHandler()
    if DEBUG:
        console_logger.setLevel(logging.DEBUG)
    else:
        console_logger.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)-4s: %(levelname)-8s %(message)s')
    console_logger.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_logger)

    LOGGER.info('All logging will be written to %s', logfile_name)

if __name__ == '__main__':
    set_up_logger()
    main()
