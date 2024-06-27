#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc
 
# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --gres gpu:2
#SBATCH --time 6-23:50:00
#SBATCH --cpus-per-gpu 9
#SBATCH --mem 80G
#SBATCH --partition longrun
#SBATCH --mail-type FAIL,END

conda activate ffyytt
python3 main_tf.py