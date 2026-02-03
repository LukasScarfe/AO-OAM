import numpy as np
import math

from LightPipes import Field
from LightPipes import BeamMix, GaussLaguerre, Phase, SubPhase


def ell(d: int) -> np.ndarray:
    """
    Generate a balanced list of size dimension _d_, such that it is centred around 0.
    
    :param d: Dimensionality of the system, length of the output array.
    :type d: int


    >>> ell(4) 
    [-2,-1,1,2]
    >>> ell(5)
    [-2,-1,0,1,2]
    """

    if d%2 == 0:
        l=np.linspace(-np.floor(d/2),np.floor(d/2),d+1,dtype=int)
        l=l[l != 0]
    else:
        l=np.linspace(-np.floor(d/2),np.floor(d/2),d,dtype=int)
    return l



def MUB_phases_basis_state(d: int,basis_choice: int=0,state_choice: int=0) -> list:

    """
    Returns the phases required to contruct a particular state in a particular mutually unbiased basis that is in a maximal set of MUB of dimension d. 
    
    :param d: Dimensionality of the system.
    :type d: int
    :param basis_choice: Choice of basis, ranges from 0 to d-1
    :type basis_choice: int
    :param state_choice: Choice of state, ranges from 0 to d-1
    :type state_choice: int
    :return: All phases for a single mode that is of dimension d, in a given Mutually Unbiased Basis
    :rtype: list

    >>> MUB_phases_basis_state(4,2,2)
    [0, 0, 1.5707963267948966, 4.71238898038469]
    >>> MUB_phases_basis_state(3,1,1)
    [0.0, 4.1887902047863905, 12.566370614359172]
    """

    # crappy little is prime function.
    def is_prime(n):
        if n ==2 or n==4 or n==8: return True
        if n == 2 or n == 3: return True
        if n < 2 or n%2 == 0: return False
        if n < 9: return True
        if n%3 == 0: return False
        r = int(n**0.5)
        f = 5
        while f <= r:
            print('\t',f)
            if n % f == 0: return False
            if n % (f+2) == 0: return False
            f += 6
        return True
    
    if not is_prime(d) :
        raise ValueError("This function is for prime numbers and 2,4,8")
    elif basis_choice not in range(d+1):
        raise ValueError(f"basis_choice needs to be in range 0 <= x <= d, given {basis_choice}")
    elif  state_choice not in range(d):
        raise ValueError(f"state_choice needs to be in range 0 <= x <= d-1, given {state_choice}")
    elif d>50:
        raise ValueError("lets keep the dimension small for now d<50")
        
    elif d==2:
        p=[[[0,0],[0,np.pi]],[[0,np.pi/2],[0,3*np.pi/2]]][basis_choice][state_choice]
        return p
    elif d==4:
        p=[[[0,0,0,0],[0,np.pi,np.pi,0],[0,0,np.pi,np.pi],[0,np.pi,0,np.pi]],[[0,np.pi/2,np.pi/2,np.pi],[0,3*np.pi/2,3*np.pi/2,np.pi],[0,np.pi/2,3*np.pi/2,0],[0,3*np.pi/2,np.pi/2,0]],[[0,0,3*np.pi/2,np.pi/2],[0,np.pi,np.pi/2,np.pi/2],[0,0,np.pi/2,3*np.pi/2],[0,np.pi,3*np.pi/2,3*np.pi/2]],[[0,3*np.pi/2,0,np.pi/2],[0,np.pi/2,np.pi,np.pi/2],[0,np.pi/2,0,3*np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2]]][basis_choice][state_choice]
        return p
    elif d==8:
        p=[[[0,0,0,0,0,0,0,0],[0,np.pi/2,np.pi/2,np.pi,np.pi/2,np.pi,np.pi,3*np.pi/2],[0,np.pi/2,0,np.pi/2,0,3*np.pi/2,np.pi,np.pi/2],[0,0,np.pi/2,np.pi/2,np.pi/2,3*np.pi/2,0,np.pi],[0,0,np.pi/2,3*np.pi/2,0,np.pi,np.pi/2,np.pi/2],[0,np.pi/2,0,3*np.pi/2,np.pi/2,0,np.pi/2,np.pi],[0,np.pi/2,np.pi/2,0,0,np.pi/2,3*np.pi/2,np.pi],[0,0,0,np.pi,np.pi/2,np.pi/2,3*np.pi/2,np.pi/2]],[[0,np.pi,0,np.pi,0,np.pi,0,np.pi],[0,3*np.pi/2,np.pi/2,0,np.pi/2,0,np.pi,np.pi/2],[0,3*np.pi/2,0,3*np.pi/2,0,np.pi/2,np.pi,3*np.pi/2],[0,np.pi,np.pi/2,3*np.pi/2,np.pi/2,np.pi/2,0,0],[0,np.pi,np.pi/2,np.pi/2,0,0,np.pi/2,3*np.pi/2],[0,3*np.pi/2,0,np.pi/2,np.pi/2,np.pi,np.pi/2,0],[0,3*np.pi/2,np.pi/2,np.pi,0,3*np.pi/2,3*np.pi/2,0],[0,np.pi,0,0,np.pi/2,3*np.pi/2,3*np.pi/2,3*np.pi/2]],[[0,0,np.pi,np.pi,0,0,np.pi,np.pi],[0,np.pi/2,3*np.pi/2,0,np.pi/2,np.pi,0,np.pi/2],[0,np.pi/2,np.pi,3*np.pi/2,0,3*np.pi/2,0,3*np.pi/2],[0,0,3*np.pi/2,3*np.pi/2,np.pi/2,3*np.pi/2,np.pi,0],[0,0,3*np.pi/2,np.pi/2,0,np.pi,3*np.pi/2,3*np.pi/2],[0,np.pi/2,np.pi,np.pi/2,np.pi/2,0,3*np.pi/2,0],[0,np.pi/2,3*np.pi/2,np.pi,0,np.pi/2,np.pi/2,0],[0,0,np.pi,0,np.pi/2,np.pi/2,np.pi/2,3*np.pi/2]],[[0,np.pi,np.pi,0,0,np.pi,np.pi,0],[0,3*np.pi/2,3*np.pi/2,np.pi,np.pi/2,0,0,3*np.pi/2],[0,3*np.pi/2,np.pi,np.pi/2,0,np.pi/2,0,np.pi/2],[0,np.pi,3*np.pi/2,np.pi/2,np.pi/2,np.pi/2,np.pi,np.pi],[0,np.pi,3*np.pi/2,3*np.pi/2,0,0,3*np.pi/2,np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2,np.pi/2,np.pi,3*np.pi/2,np.pi],[0,3*np.pi/2,3*np.pi/2,0,0,3*np.pi/2,np.pi/2,np.pi],[0,np.pi,np.pi,np.pi,np.pi/2,3*np.pi/2,np.pi/2,np.pi/2]],[[0,0,0,0,np.pi,np.pi,np.pi,np.pi],[0,np.pi/2,np.pi/2,np.pi,3*np.pi/2,0,0,np.pi/2],[0,np.pi/2,0,np.pi/2,np.pi,np.pi/2,0,3*np.pi/2],[0,0,np.pi/2,np.pi/2,3*np.pi/2,np.pi/2,np.pi,0],[0,0,np.pi/2,3*np.pi/2,np.pi,0,3*np.pi/2,3*np.pi/2],[0,np.pi/2,0,3*np.pi/2,3*np.pi/2,np.pi,3*np.pi/2,0],[0,np.pi/2,np.pi/2,0,np.pi,3*np.pi/2,np.pi/2,0],[0,0,0,np.pi,3*np.pi/2,3*np.pi/2,np.pi/2,3*np.pi/2]],[[0,np.pi,0,np.pi,np.pi,0,np.pi,0],[0,3*np.pi/2,np.pi/2,0,3*np.pi/2,np.pi,0,3*np.pi/2],[0,3*np.pi/2,0,3*np.pi/2,np.pi,3*np.pi/2,0,np.pi/2],[0,np.pi,np.pi/2,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi,np.pi],[0,np.pi,np.pi/2,np.pi/2,np.pi,np.pi,3*np.pi/2,np.pi/2],[0,3*np.pi/2,0,np.pi/2,3*np.pi/2,0,3*np.pi/2,np.pi],[0,3*np.pi/2,np.pi/2,np.pi,np.pi,np.pi/2,np.pi/2,np.pi],[0,np.pi,0,0,3*np.pi/2,np.pi/2,np.pi/2,np.pi/2]],[[0,0,np.pi,np.pi,np.pi,np.pi,0,0],[0,np.pi/2,3*np.pi/2,0,3*np.pi/2,0,np.pi,3*np.pi/2],[0,np.pi/2,np.pi,3*np.pi/2,np.pi,np.pi/2,np.pi,np.pi/2],[0,0,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi/2,0,np.pi],[0,0,3*np.pi/2,np.pi/2,np.pi,0,np.pi/2,np.pi/2],[0,np.pi/2,np.pi,np.pi/2,3*np.pi/2,np.pi,np.pi/2,np.pi],[0,np.pi/2,3*np.pi/2,np.pi,np.pi,3*np.pi/2,3*np.pi/2,np.pi],[0,0,np.pi,0,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi/2]],[[0,np.pi,np.pi,0,np.pi,0,0,np.pi],[0,3*np.pi/2,3*np.pi/2,np.pi,3*np.pi/2,np.pi,np.pi,np.pi/2],[0,3*np.pi/2,np.pi,np.pi/2,np.pi,3*np.pi/2,np.pi,3*np.pi/2],[0,np.pi,3*np.pi/2,np.pi/2,3*np.pi/2,3*np.pi/2,0,0],[0,np.pi,3*np.pi/2,3*np.pi/2,np.pi,np.pi,np.pi/2,3*np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2,3*np.pi/2,0,np.pi/2,0],[0,3*np.pi/2,3*np.pi/2,0,np.pi,np.pi/2,3*np.pi/2,0],[0,np.pi,np.pi,np.pi,3*np.pi/2,np.pi/2,3*np.pi/2,3*np.pi/2]]][state_choice][basis_choice]
        return p
    
    else: return [2* np.pi/d * (basis_choice*l**2+state_choice*l) for l in range(d)]


def MUB_phases_basis(d: int,basis_choice: int=0) -> list[list]:
    """
    Returns the phases required to contruct all states in a particular mutually unbiased basis that is in a maximal set of MUB for the given dimension. 
    
    :param d: Dimensionality of the system.
    :type d: int
    :param basis_choice: Choice of basis, ranges from 0 to d-1
    :type basis_choice: int
    :return: list of all phases required to make the states that are in the given basis choice.
    :rtype: list[list]

    >>> MUB_phases_basis(2,0)
    [[0, 0], [0, 3.141592653589793]]
    >>> MUB_phases_basis(3,1)

    [[0.0, 2.0943951023931953, 8.377580409572781], [0.0, 4.1887902047863905, 12.566370614359172], [0.0, 6.283185307179586, 16.755160819145562]]
    """

    if basis_choice not in range(d):
        raise ValueError(f"basis_choice needs to be in range 0 <= x <= d, given {basis_choice}")
    return [MUB_phases_basis_state(d,basis_choice,state_choice=l) for l in range(d)]


#All MUBS in a basis
def MUB_phases(d: int) -> list[list[list]]:
    """
    Docstring for MUB_phases
    
    :param d: Dimensionality of the system.
    :type d: int
    :return: All phases required to make all states for a given dimension's maximal set of mutually unbiased bases. 
    :rtype: list[list[list]]

    >>> MUB_phases(2)
    [[0, 0], [0, 3.141592653589793], [0, 1.5707963267948966], [0, 4.71238898038469]]

    >>> MUB_phases(3)
    [[0.0, 0.0, 0.0], [0.0, 2.0943951023931953, 4.1887902047863905], [0.0, 4.1887902047863905, 8.377580409572781], [0.0, 2.0943951023931953, 8.377580409572781], [0.0, 4.1887902047863905, 12.566370614359172], [0.0, 6.283185307179586, 16.755160819145562], [0.0, 4.1887902047863905, 16.755160819145562], [0.0, 6.283185307179586, 20.94395102393195], [0.0, 8.377580409572781, 25.132741228718345]]
    """
    return [MUB_phases_basis_state(d,basis_choice=b,state_choice=l) for b in range(d) for l in range(d)]


#Mixing large amount of beams in a loop
def mix_many_beams(beams_list: list[Field]) -> Field:
    """
    Mix many LightPipes fields in a single function.
    
    :param beams: list of LightPipes fields to be mixed.
    :type beams: list[Field]
    :return: Returns the mixed fields in a single field
    :rtype: Field
    """
    if not isinstance(beams_list, list): raise ValueError("input fields must be a list of Fields")
    elif len(beams_list) > 1:
        beams_list[1]=BeamMix(beams_list[0],beams_list[1])
        beams_list.pop(0)
        mix_many_beams(beams_list)
    return beams_list[0]


def OAM(F: Field, beam_radius: float, state: int=0, phase: float=0, amp: float=1.0) -> Field:
    """
    Create an OAM beam LightPipes Field. 
    
    :param F: Input LightPipes field.
    :type F: Field
    :param beam_radius: radius of the beam waist.
    :type beam_radius: float
    :param state: l value for OAM.
    :type state: int
    :param phase: Additional fixed relative phase component to add to the beam, used for making optical modes which are superpositions with relative phases. 
    :type phase: float
    :param amp: amplitude of the OAM beam.
    :type amp: float
    :return: OAM mode output.
    :rtype: Field
    """

    F=GaussLaguerre(F, beam_radius, p=0, l=state, A=amp*(1/beam_radius)*np.sqrt(2/(np.pi*(math.factorial(abs(state))))), ecs=0)
    F=SubPhase(F,Phase(F)+phase)
    return F

#Arbitrary ANG mode in dimension d
def ANG(F: Field, beam_radius: float ,d: int,state: int) -> Field:
    """
    Generates the angular mode LightPipes Field, which is the conjugate basis of the orbital angular momentum modes. OAM is angular momentum, while the Angular Modes are the corresponding angular position modes, similar to how cartesian position and momentum are conjugate.
    
    :param F: Input LightPipes field.
    :type F: Field
    :param beam_radius: radius of the beam waist.
    :type beam_radius: float
    :param d: Dimensionality of the system.
    :type d: int
    :param state: Choice of state, ranges from 0 to d-1
    :type state: int
    :return: output Angular mode as a LightPipes Field.
    :rtype: Field
    """

    if  state not in range(d):
        raise ValueError(f"state_choice needs to be in range 0 <= x <= d-1, given {state}")
    
    Q,p=[],[]
    l=ell(d)
    
    for i in range(d):
        p.append((2*np.pi/d)*i*state)

    for i in range(d):
        Q.append(OAM(F,beam_radius,l[i],p[i]))

    F=mix_many_beams(Q)

    return F
    

# Not importand right now and needs a complete rework
# def ArbMUB(F,w0,d,MUB,state,norm=0):
#     intensityNorm=[1,1,187.3568368182197,302.7415771427128,281.4774347094998,342.95046333063704,1,330.2491174614896,276.25225696328897,1,1,1,1,1,1,1,1,1,1,1,1,1]
#     Q,p=[],[]
#     l=ell(d)
#     if d==2:
#         p=[[[0,0],[0,np.pi]],[[0,np.pi/2],[0,3*np.pi/2]]][MUB][state]
#     elif d==4:
#         p=[[[0,0,0,0],[0,np.pi,np.pi,0],[0,0,np.pi,np.pi],[0,np.pi,0,np.pi]],[[0,np.pi/2,np.pi/2,np.pi],[0,3*np.pi/2,3*np.pi/2,np.pi],[0,np.pi/2,3*np.pi/2,0],[0,3*np.pi/2,np.pi/2,0]],[[0,0,3*np.pi/2,np.pi/2],[0,np.pi,np.pi/2,np.pi/2],[0,0,np.pi/2,3*np.pi/2],[0,np.pi,3*np.pi/2,3*np.pi/2]],[[0,3*np.pi/2,0,np.pi/2],[0,np.pi/2,np.pi,np.pi/2],[0,np.pi/2,0,3*np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2]]][MUB][state]
#     elif d==8:
#         p=[[[0,0,0,0,0,0,0,0],[0,np.pi/2,np.pi/2,np.pi,np.pi/2,np.pi,np.pi,3*np.pi/2],[0,np.pi/2,0,np.pi/2,0,3*np.pi/2,np.pi,np.pi/2],[0,0,np.pi/2,np.pi/2,np.pi/2,3*np.pi/2,0,np.pi],[0,0,np.pi/2,3*np.pi/2,0,np.pi,np.pi/2,np.pi/2],[0,np.pi/2,0,3*np.pi/2,np.pi/2,0,np.pi/2,np.pi],[0,np.pi/2,np.pi/2,0,0,np.pi/2,3*np.pi/2,np.pi],[0,0,0,np.pi,np.pi/2,np.pi/2,3*np.pi/2,np.pi/2]],[[0,np.pi,0,np.pi,0,np.pi,0,np.pi],[0,3*np.pi/2,np.pi/2,0,np.pi/2,0,np.pi,np.pi/2],[0,3*np.pi/2,0,3*np.pi/2,0,np.pi/2,np.pi,3*np.pi/2],[0,np.pi,np.pi/2,3*np.pi/2,np.pi/2,np.pi/2,0,0],[0,np.pi,np.pi/2,np.pi/2,0,0,np.pi/2,3*np.pi/2],[0,3*np.pi/2,0,np.pi/2,np.pi/2,np.pi,np.pi/2,0],[0,3*np.pi/2,np.pi/2,np.pi,0,3*np.pi/2,3*np.pi/2,0],[0,np.pi,0,0,np.pi/2,3*np.pi/2,3*np.pi/2,3*np.pi/2]],[[0,0,np.pi,np.pi,0,0,np.pi,np.pi],[0,np.pi/2,3*np.pi/2,0,np.pi/2,np.pi,0,np.pi/2],[0,np.pi/2,np.pi,3*np.pi/2,0,3*np.pi/2,0,3*np.pi/2],[0,0,3*np.pi/2,3*np.pi/2,np.pi/2,3*np.pi/2,np.pi,0],[0,0,3*np.pi/2,np.pi/2,0,np.pi,3*np.pi/2,3*np.pi/2],[0,np.pi/2,np.pi,np.pi/2,np.pi/2,0,3*np.pi/2,0],[0,np.pi/2,3*np.pi/2,np.pi,0,np.pi/2,np.pi/2,0],[0,0,np.pi,0,np.pi/2,np.pi/2,np.pi/2,3*np.pi/2]],[[0,np.pi,np.pi,0,0,np.pi,np.pi,0],[0,3*np.pi/2,3*np.pi/2,np.pi,np.pi/2,0,0,3*np.pi/2],[0,3*np.pi/2,np.pi,np.pi/2,0,np.pi/2,0,np.pi/2],[0,np.pi,3*np.pi/2,np.pi/2,np.pi/2,np.pi/2,np.pi,np.pi],[0,np.pi,3*np.pi/2,3*np.pi/2,0,0,3*np.pi/2,np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2,np.pi/2,np.pi,3*np.pi/2,np.pi],[0,3*np.pi/2,3*np.pi/2,0,0,3*np.pi/2,np.pi/2,np.pi],[0,np.pi,np.pi,np.pi,np.pi/2,3*np.pi/2,np.pi/2,np.pi/2]],[[0,0,0,0,np.pi,np.pi,np.pi,np.pi],[0,np.pi/2,np.pi/2,np.pi,3*np.pi/2,0,0,np.pi/2],[0,np.pi/2,0,np.pi/2,np.pi,np.pi/2,0,3*np.pi/2],[0,0,np.pi/2,np.pi/2,3*np.pi/2,np.pi/2,np.pi,0],[0,0,np.pi/2,3*np.pi/2,np.pi,0,3*np.pi/2,3*np.pi/2],[0,np.pi/2,0,3*np.pi/2,3*np.pi/2,np.pi,3*np.pi/2,0],[0,np.pi/2,np.pi/2,0,np.pi,3*np.pi/2,np.pi/2,0],[0,0,0,np.pi,3*np.pi/2,3*np.pi/2,np.pi/2,3*np.pi/2]],[[0,np.pi,0,np.pi,np.pi,0,np.pi,0],[0,3*np.pi/2,np.pi/2,0,3*np.pi/2,np.pi,0,3*np.pi/2],[0,3*np.pi/2,0,3*np.pi/2,np.pi,3*np.pi/2,0,np.pi/2],[0,np.pi,np.pi/2,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi,np.pi],[0,np.pi,np.pi/2,np.pi/2,np.pi,np.pi,3*np.pi/2,np.pi/2],[0,3*np.pi/2,0,np.pi/2,3*np.pi/2,0,3*np.pi/2,np.pi],[0,3*np.pi/2,np.pi/2,np.pi,np.pi,np.pi/2,np.pi/2,np.pi],[0,np.pi,0,0,3*np.pi/2,np.pi/2,np.pi/2,np.pi/2]],[[0,0,np.pi,np.pi,np.pi,np.pi,0,0],[0,np.pi/2,3*np.pi/2,0,3*np.pi/2,0,np.pi,3*np.pi/2],[0,np.pi/2,np.pi,3*np.pi/2,np.pi,np.pi/2,np.pi,np.pi/2],[0,0,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi/2,0,np.pi],[0,0,3*np.pi/2,np.pi/2,np.pi,0,np.pi/2,np.pi/2],[0,np.pi/2,np.pi,np.pi/2,3*np.pi/2,np.pi,np.pi/2,np.pi],[0,np.pi/2,3*np.pi/2,np.pi,np.pi,3*np.pi/2,3*np.pi/2,np.pi],[0,0,np.pi,0,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi/2]],[[0,np.pi,np.pi,0,np.pi,0,0,np.pi],[0,3*np.pi/2,3*np.pi/2,np.pi,3*np.pi/2,np.pi,np.pi,np.pi/2],[0,3*np.pi/2,np.pi,np.pi/2,np.pi,3*np.pi/2,np.pi,3*np.pi/2],[0,np.pi,3*np.pi/2,np.pi/2,3*np.pi/2,3*np.pi/2,0,0],[0,np.pi,3*np.pi/2,3*np.pi/2,np.pi,np.pi,np.pi/2,3*np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2,3*np.pi/2,0,np.pi/2,0],[0,3*np.pi/2,3*np.pi/2,0,np.pi,np.pi/2,3*np.pi/2,0],[0,np.pi,np.pi,np.pi,3*np.pi/2,np.pi/2,3*np.pi/2,3*np.pi/2]]][state][MUB]
#     else:
#         for j in range(d):
#             p.append(((2*np.pi/d)*(MUB*(j**2)+state*j)))
            
#     for i in range(d):
#         Q.append(OAM(F,w0,l[i],p[i]))
#     F=MixManyBeams(Q)

#     if norm==1:
#         return [F,intensityNorm[d]]
#     else:
#         return F


def Fqubit(F: Field,beam_radius: float,d: int,j: int=0,k: int=1,m: int=0) -> Field:
    """
    Generates the Fourier Qubit mode for the given input parameters. (See https://r.lukasscarfe.com/fqb)
    
    :param F: Input LightPipes field.
    :type F: Field
    :param beam_radius: radius of the beam waist.
    :type beam_radius: float
    :param d: Dimensionality of the system.
    :type d: int
    :param j: j parameter of the Fourier Qubit mode
    :type j: int
    :param k: k parameter of the Fourier Qubit mode
    :type k: int
    :param m: m parameter of the Fourier Qubit mode
    :type m: int
    :return: output F-Qubit mode as a LightPipes Field.
    :rtype: Field
    """
    l=ell(d)
    p=(2*np.pi/d)*m

    if j<0 or j>=k or j>d-1 or k<1 or k>d or m<0 or m>d:
        return ValueError("see rules for the F-qubit parameters: j -> [0,d-1], k-> [j+1,d-1], m -> [0,d-1], (See https://r.lukasscarfe.com/fqb)")
    else:
        Q=BeamMix(OAM(F,beam_radius,l[j],0),OAM(F,beam_radius,l[k],p))
    return Q


def wrap_to_pi(angle: float) -> float:
    """
    Forces the phases to be normalized back between -pi and pi
    
    :param angle: input angle to be normed within 0 and pi
    :type angle: float
    :return: normalized angle
    :rtype: float

    >>> wrap_to_pi(4*np.pi)
    0.0
    >>> wrap_to_pi(4/3*np.pi)
    -2.0943951023931957
    """
    return (angle + np.pi) % (2 *np.pi) - np.pi
