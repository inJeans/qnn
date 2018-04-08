import platform
import os
import datetime
import logging

from scipy.linalg import expm
from pyquil.quil import Program
from pyquil.quilbase import DefGate
from pyquil.api import QVMConnection
from pyquil.gates import CNOT, X, TRUE, FALSE, NOT, RZ, RY
from pyquil.paulis import sY, exponential_map
from pyquil.parameters import Parameter, quil_sin, quil_cos

import numpy as np

LOGGER = logging.getLogger("qnn")
DEBUG = False

ANCILLARY_BIT = 1
OUTPUT_BIT = ANCILLARY_BIT + 1

def main():
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

    a = 1
    rotation = 0.5*np.pi*(a + 1)
    theta = 0.3 * np.pi

    p.inst(RY(rotation, 0))
    # p.inst(CRY(theta)(0, 1)).measure(0, 0).measure(1,1)
    # LOGGER.info("... %s", p)

    classical_flag_register = 3
    # Write out the loop initialization and body programs:
    loop_body = Program(CRY(2.*theta)(0, ANCILLARY_BIT))
    loop_body.inst(CSY(ANCILLARY_BIT, OUTPUT_BIT))
    loop_body.inst(RZ(-0.5*np.pi)(ANCILLARY_BIT))
    loop_body.inst(CRY(-2.*theta)(0, ANCILLARY_BIT))

    # print(qvm.wavefunction(loop_body))

    loop_body.measure(ANCILLARY_BIT, classical_flag_register)
    
    then_branch = Program(RY(-0.5*np.pi)(OUTPUT_BIT))
    then_branch.inst(X(ANCILLARY_BIT))
    else_branch = Program()

    # # Add the conditional branching:
    loop_body.if_then(classical_flag_register,
                      then_branch,
                      else_branch)

    init_register = Program(TRUE([classical_flag_register]))
    loop_prog = init_register.while_do(classical_flag_register,
                                       loop_body)
    p.inst(loop_prog)

    p.measure(0,0).measure(1,1).measure(2,2)

    LOGGER.info("... executing on the QVM")
    classical_regs = [0, 1, 2, 3]
    output = qvm.run(p, classical_regs)
    LOGGER.info("... %s", output)
    LOGGER.info("")

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
