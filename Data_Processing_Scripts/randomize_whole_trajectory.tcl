#!/usr/bin/env vmd
############################################################## 
# Author:               John Vant 
# Email:              jvant@asu.edu 
# Affiliation:   ASU Biodesign Institute 
# Date Created:          210218
############################################################## 
# Usage: See end of the script for usage examples
############################################################## 
# Notes: You can use these functions on DCD or PDB inputs
############################################################## 
​
proc moveCOMtoOrigin { sel } {
    set currentCOM [measure center $sel]
    set myVec [list ]
    for { set i 0 } { $i < 3 } { incr i } {
	lappend myVec [expr { 0.0 - [lindex $currentCOM $i] }]	    
    }
    set tansMatrix "{ 1.0 0.0 0.0 [lindex $myVec 0]} \
	{0.0 1.0 0.0 [lindex $myVec 1]} \
	{0.0 0.0 1.0 [lindex $myVec 2]} \
	{0.0 0.0 0.0 1.0}"
    $sel move $tansMatrix
    return
}
​
proc sameple_random_quaternion { } {
    # Sample 3 random numbers
    set rand1 [expr rand()]; set rand2 [expr rand()]; set rand3 [expr rand()]
    # Calc Euler angles
    set myPhi [ expr $rand1 * 2 * 3.14159 ]
    set myTheta [ expr $rand2 * 2 * 3.14159 ]
    set myZ [ expr $rand3 * 2 ]
    # Calc transition matrix
    set myR [expr { sqrt( $myZ ) }]
    set myVx [expr { sin($myPhi) * $myR }]
    set myVy [expr { cos($myPhi) * $myR }]
    set myVz [expr { sqrt(2.0 - $myZ) }]
    set myST [expr sin($myTheta)]
    set myCT [expr cos($myTheta)]
    set mySx [expr $myVx * $myCT - $myVy * $myST]
    set mySy [expr $myVx * $myST + $myVy * $myCT]
    # Matrix elements
    set a11 [expr $myVx * $mySx - $myCT]
    set a12 [expr $myVx * $mySy - $myST]
    set a13 [expr $myVx * $myVz]
    set a21 [expr $myVy * $mySx + $myST]
    set a22 [expr $myVy * $mySy - $myCT]
    set a23 [expr $myVy * $myVz]
    set a31 [expr $myVz * $mySx]
    set a32 [expr $myVz * $mySy]
    set a33 [expr 1.0 - $myZ]
    set tansMatrix "\
    	{$a11 $a12 $a13 1.0} \
    	{$a21 $a22 $a23 1.0} \
	{$a31 $a32 $a33 1.0} \
	{0.0 0.0 0.0 1.0}"
    puts "My transistion matrix:\n$tansMatrix"
    return $tansMatrix
}    
​
proc randomize_trajectory { psffile dcdfile pdbOutpuBaseName } { 
    mol new $psffile;		# Load pdb
    mol addfile $dcdfile
    set sel [atomselect $myMolID all]
    moveCOMtoOrigin $sel
    set myMolID [molinfo top]
    set nf [molinfo $myMolID get numframes]
    set my_quaternion [sameple_random_quaternion]
    for {set i 0} {$i < $nf} {incr i} {
	$sel frame $i
	moveCOMtoOrigin $sel
	$sel move $my_quaternion
	$sel writepdb $pdbOutpuBaseName-$i.pdb;    # Write new pdb
    }
    return
}
​
# # Example pdb implementation
# for {set i 0} {$i < 100} {incr i} {
#     randomize_orientation_pdb my-$i.pdb ./output/rand-orient-$i.pdb
# }
​
# # Example dcd implementation
#     randomize_orientation_dcd example.psf example.dcd ./example/output/rand-orient



# exit
