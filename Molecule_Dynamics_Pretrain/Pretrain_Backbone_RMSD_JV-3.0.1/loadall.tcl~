mol default style Licorice

set sys ./
mol new $sys/backbone.psf
for {set i 1} {$i <= 8} {incr i} {
    mol addfile $sys/Pretrain.xyz waitfor -1
}
set myid [molinfo top]
# Rename
mol rename $myid $sys
#mol modstyle 0 $myid VDW 1.000000 12.000000
mol modmaterial 0 $myid Opaque
mol modcolor 0 $myid Name
mol modselect 0 $myid all
# Add Rep
# mol color Name
# mol representation Licorice 0.300000 12.000000 12.000000
# mol selection "segname V2"
# mol material Opaque
# mol addrep $myid
