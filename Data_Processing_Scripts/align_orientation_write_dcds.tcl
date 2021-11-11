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


proc alignCOM { refmolid refframe mymolid seltext } { 
    set refselfit [atomselect $refmolid $seltext frame $refframe]
    moveCOMtoOrigin $refselfit
    set myselfit [atomselect $mymolid $seltext]
    set myselall [atomselect $mymolid all]
    set nf [molinfo $mymolid get numframes]
    for {set i 0} {$i < $nf} {incr i} {
	$refselfit frame $refframe
	$myselall frame $i
	$myselfit frame $i
	$myselall move [measure fit $myselfit $refselfit]
    }
    $myselall delete
    $myselfit delete
    $refselfit delete
    return
}


for {set k 0} {$k < 1} {incr k} {
    for {set replica 0} {$replica < 50} {incr replica} {
	# # Example dcd implementation
	puts $replica
	mol new ../../Build/da.psf
	set id [mol addfile $replica/smd_out.dcd waitfor -1]
	alignCOM $id 0 $id "backbone"
	set selwrite [atomselect $id "protein"]
	$selwrite writepsf $replica/backbone_new.psf
	animate write dcd $replica/smd_aligned.dcd waitfor all sel $selwrite $id
	mol delete $id
    }
}



exit
