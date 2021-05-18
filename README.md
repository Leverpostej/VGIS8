# VGIS8
This reposity contains the code and files used for our 8th semester VGIS project at Aalborg University. 

The folders FAWDN, BlindSR, DualSR and EnsembleLearing-Network contains current SOTA SR solutions, described and tested in the report. 
The folder datasets contains different datasets used over the course of the project; MRI13 being the most prominent one. 

Ensemble.py is the main script responsible for creating ensembles. 
To use:
1. Download the folder with the models from [Drive](https://drive.google.com/drive/folders/1oX7MLvJERDgf1UUrnDb2w33lqfJmmmJf?usp=sharing)
2. Put the folder in VGIS8\FAWDN\options
3. Open the script in your prefered IDE (tested in Pycharm)
4. Change lines 201-203 with your data 
5. Run the script. 


Individual model results pop up and are saved in VGIS8\FAWDN\results\Modelpics.
Ensemble SR results pop up and are saved in VGIS8\FAWDN\results\Ensemble.
