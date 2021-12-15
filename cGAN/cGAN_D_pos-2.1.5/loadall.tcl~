mol default style Licorice

set sys ./
set sysname "pretrain_generated"
mol new $sys/da.psf
mol addfile $sys/pretrain_generated.xyz waitfor -1
set myid [molinfo top]
# Rename
mol rename $myid $sysname
mol modmaterial 0 $myid Opaque
mol modcolor 0 $myid Name
mol modselect 0 $myid all
 
## Genergated by GAN
set sysname "cGAN_generated"
mol new $sys/da.psf
mol addfile $sys/cGAN_generated.xyz waitfor -1
set myid [molinfo top]
# Rename
mol rename $myid $sysname
mol modmaterial 0 $myid Opaque
mol modcolor 0 $myid Name
mol modselect 0 $myid all


# Pretraining Training data set
set sysname "pretrain_train_data"
mol new $sys/da.psf
mol addfile $sys/pretrain_training_data.xyz waitfor -1
set myid [molinfo top]
# Rename
mol rename $myid $sysname
mol modmaterial 0 $myid Opaque
mol modcolor 0 $myid Name
mol modselect 0 $myid all

