#!/bin/bash
####
# jobpip.sh to deploy jobs in a folder
####

start_chunk=0
end_chunk=1

parent_folder_name="Mol-HNN-cuda-v1"

## enter the folder of interest
cd $parent_folder_name
cd train/
submit_jobid=""
jobid=""

for curr_chunk in $(eval echo {${start_chunk}..${end_chunk}});
do
        if [ -z $jobid ];then
            #jobid=$(mysbatch equil-runnamd)
                submit_jobid=$(sbatch run${curr_chunk}.sh)
                jobid="${submit_jobid//[!0-9]/}"
            echo "First submission: ${jobid} deployed for ${curr_chunk} for ${parent_folder_name}"
        else
            #jobid=$(mysbatch -d afterany:$jobid equil-runnamd)
                submit_jobid=$(sbatch -d afterany:$jobid run${curr_chunk}.sh)
                jobid="${submit_jobid//[!0-9]/}"
            echo "Subsequent Submissions out${k}: ${jobid}"
        fi
done

cd ../
    
cd ../

exit
