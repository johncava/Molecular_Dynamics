#!/bin/bash

# Functions
mysbatch () {
    sbatch \
	--parsable \
	-N $Nodes \
	-n $Cores \
	-t $Time \
	-o slurm_${cmstr}.log \
	-p $Partition -q $Queue \
	-J ${cmstr} \
	$@
}
# Jobname Argument
if [ -z "$1" ]; then
    echo "Please enter a unique job name";read jobname
else
    jobname=$1
fi

# Set Defaults
Nodes=1
Cores=4
Gpus=1 # not used
Partition=htc
Queue=normal
Time=4:00:00
threads=$(expr $Cores - $Gpus)

cd output

for replica in {20..49}; do
    echo "Replica $replica"
    cd $replica    
	cmstr=$jobname-output
	# NAMD
	cat <<EOF > deca-runnamd
#!/bin/bash
module load namd/2.13-mpi
namd2 smd.namd
EOF

echo "test"
jobid=$(mysbatch deca-runnamd)
echo "Job named " $cmstr ", submitted with JOBID: " $jobid
cd ..

done

