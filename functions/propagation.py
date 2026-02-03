import os

from LightPipes import Field
from LightPipes import Forvard, Forward, Fresnel, Phase, SubPhase, CircAperture, SubIntensity

import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm as progress

### Propagation
#Propagates one beam through the channel defined by the distance and the list of abberations that is input (abbs)
#If there are abberations, distance is the distance per abberation
def propChannel(F: Field,distance: float, abbs=1,mode=0):
    if not hasattr(abbs, "__len__"):
        for i in range(abbs):
            if mode==1:
                F=Forvard(F,distance/abbs)
            elif mode == 2:
                F=Forward(F,distance/abbs,F.siz,F.N)
            else:
                F=Fresnel(F,distance/abbs)

    else:
        F=Fresnel(F,distance/(len(abbs)+1))
        for screen in abbs:
            if not isinstance(screen, np.ndarray):
                screen_real = screen.scrn.copy() # needed for infinite phase screens
            else: 
                screen_real = screen
            F=SubPhase(F,Phase(F)+screen_real)
            if mode==1:
                F=Forvard(F,distance/(len(abbs)+1))
            elif mode == 2:
                F=Forward(F,distance/(len(abbs)+1),F.size,F.N)
            else:
                F=Fresnel(F,distance/(len(abbs)+1))
    return F

def propChannelSteps(F,distance,abbs=1):
    Fs=[]
    if not hasattr(abbs, "__len__"):
        for i in abbs:
            F=Forvard(F,int(distance/abbs))
            Fs.append(F)
            F=Forvard(F,distance)
    else:
        for screen in abbs:
            F=SubPhase(F,Phase(F)+screen)
            F=Forvard(F,int(distance/len(abbs)))
            Fs.append(F)
    return ([F,Fs])

### Parellelization of propagation of all pixels

def propagateSinglePixel(FieldIn: Field,j: int, i:int, N:int, z:float, lensSize:float, abbs) -> tuple:
    """
    Performs the calculation for a single (j, i) pair.

    Args:
        FieldIn: Initial field.
        j (int): Outer loop index.
        i (int): Inner loop index.
        N (int): Grid size.
        z (float): Propagation distance.
        abbs: List of numpy ndarray representing phase screens.

    Returns:
        tuple: (FieldOut, endField_data) for the specific (j, i)
    """   
    
    # 1. Create the intensity matrix
    intens = np.zeros((N, N))
    intens[j][i] = 1

    fieldAtSender=CircAperture(SubPhase(SubIntensity(FieldIn, intens),np.zeros((N,N))),lensSize)

    
    #FieldOut = propChannel(SubIntensity(FieldIn, intens), z, abbs)
    #FieldOut = propChannel(CircAperture(SubIntensity(FieldIn, intens),lensSize), z, abbs)
    FieldOut = CircAperture(propChannel(fieldAtSender, z, abbs),lensSize)

    # 4. Extract data
    endField_data = np.array(FieldOut.field).reshape(N**2)

    return FieldOut,endField_data

def parallelpropagatePixels(FieldIn, N, z,lensSize, abbs):
        
    # Initialize the results lists
    FieldsOut = []
    endFields_data = []
    
    max_workers = os.cpu_count()-2 or 4
    print(f"Using {max_workers} threads to propagate beams simulataneously...")

    tasks = [(j, i) for j in range(N) for i in range(N)]
    ...
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            # Pass the raw data/args, NOT the complex FieldIn object
            executor.submit(propagateSinglePixel, FieldIn, j, i, N, z,lensSize, abbs): (j, i) 
            for j, i in tasks
        }
        
        # Iterate over the results as they complete, showing progress
        results_iterator = progress(
            as_completed(future_to_task), 
            total=len(tasks), 
            desc="Propagating invidual Pixels"
        )
        
        for future in results_iterator:
            try:
                # Get the result from the completed task
                FieldOut,endField_data = future.result()
                
                # Append the results to the main lists
                FieldsOut.append(FieldOut)
                endFields_data.append(endField_data)
                
            except Exception as exc:
                # Print any exceptions that occurred inside the worker function
                j, i = future_to_task[future]
                print(f'\n(j={j}, i={i}) generated an exception: {exc}')
                
    return FieldsOut,np.array(endFields_data)



def extend_phase_screen(screen, direction="down", num_steps=1):

    def add_row_down(screen, num_steps=1):
        return 1

    def add_row_left(screen, num_steps=1):
        return 2
    
    def add_row_up(screen, num_steps=1):
        return 3
    
    def add_row_right(screen, num_steps=1):
        return 4

    if direction == "down" or 0:
        return add_row_down(screen, num_steps=1)
    
    if direction == "left" or 1:
        return add_row_left(screen, num_steps=1)
    
    if direction == "up" or 2:
        return add_row_up(screen, num_steps=1)
    
    if direction == "right" or 3:
        return add_row_right(screen, num_steps=1)
