#!/usr/bin/env python
'''
############################################################## 
# Author:               John Vant 
# Email:              jvant@asu.edu 
# Affiliation:   ASU Biodesign Institute 
# Date Created:          210905
############################################################## 
# Usage: 
############################################################## 
# Notes: 
############################################################## 
'''
import os


def mymkdir(s):
    if not os.path.exists(s):
        os.makedirs(s)


def mysymlink(source, dest):
    if not os.path.exists(dest):
        os.symlink(source, dest)


def createdirectory(replica):
    mymkdir("output")
    os.chdir("output")
    mymkdir(str(replica))
    os.chdir(str(replica))
    mysymlink("../../../Build/da.psf", "my.psf")
    mysymlink("../../../Build/smd_ini.pdb", "my.pdb")
    mysymlink("../../../charmm", "charmm")
    fout = open("smd.namd", "w")
    fout.write(SCRIPT % COLVAR_HARMONIC)
    fout.close()
    os.chdir("../../")



SCRIPT = '''
set molname my
set num_steps 500000    ;# 1 ns
set temperature 300.0          
set logfreq 500         
set dcdfreq 50          ;# 10,000 frames given 500000 steps, or 1 frame every 100 fs
#restarts cost almost nothing for small systems
set restartfreq 25000

# Input
structure   $molname.psf
coordinates $molname.pdb

paraTypeCharmm          on
parameters charmm/par_all27_prot_lipid_cmap.prm

#Force field modifications
exclude scaled1-4
1-4scaling 1.0
dielectric 1.0
gbis                on
switching           on
VDWForceSwitching   on
alphacutoff         14.
switchdist          15.
cutoff              16.
pairlistdist        17.
ionconcentration    0.1
solventDielectric   80.0
sasa                on
stepspercycle       20
margin              2.0
rigidBonds          ALL
timestep            2.0

#Thermostat. I always use a damping coefficient of 1, but that might be my membrane bias.
langevin on
temperature $temperature
langevinTemp $temperature
langevinDamping 1.0
langevinHydrogen no

# Standard output frequencies
outputEnergies          $logfreq
outputTiming            $logfreq
DCDFreq                 $dcdfreq
restartfreq             $restartfreq


#Set outputname, GRIDFILE, and which colvar to operate with.
outputname smd_out
colvars on
cv config "
colvarsTrajFrequency $dcdfreq
%s
"

minimize 500
run $num_steps

'''


COLVAR_HARMONIC = '''
colvar {
	# difference of two rmsd's
	name End2EndDist
        distance {
        group1 {
                psfSegID BH
                atomNameResidueRange CA 1-1
        }
        group2 {
                psfSegID BH
                atomNameResidueRange CA 10-10
	}
        }
}

harmonic {
	colvars                 End2EndDist
	centers                 12.0
	targetCenters           34.0
	targetNumSteps          $num_steps  				;# 1 ns
	forceConstant           1.0
        outputCenters           yes
	outputAccumulatedWork   yes
}

'''


# Main
for rep in range(50):
        createdirectory(rep)

exit()
