## Categorization
Categorization contains methods to calculate clusterings and visualize results for Atari Games based Data.

## Usage
To use implemented methods, simply create your own Python Program in the directory and import AtariGamesClusterings.py. To use implemented methods, simply instantiate an object of the class with [Data](#data).

```python
import AtariGamesClustering.AtariGamesClustering

#data must be a .csv file in ./CategData following
data = "FILENAME"
obj = AtariGamesClustering(data)
```
All following sections assume that a valid object of AtariGamesClustering was created in the beginning.
## Data
To cluster Atari games into categories, a .csv file in ./CatData needs to be provided. Data should have the following format:  
| Games | Feature_1 | Feature_2 | ... | Feature_m
| :---: | :---: | :---: | :---: | :---: | 
| GameName_1 | x_1 | y_1 | ... | z_1
| GameName_2 | x_2 | y_2 | ... | z_1  
| ... |  ... | ... | ... | ...
| GameName_n| x_n | y_n | ... | z_n

Where the column "Games" should contain for all clustered games the corresponding name of the game, for example "Pong". For each feature that is used in the clustering, a corresponding column is created with the data.  
Ideally, the naming scheme explained in ./CatData/info.txt should be followed, as it allows for compatibility checking when using [baseline normalization](#Baseline) for Deep Reinforcement Learning agent scores as clustering data.    
## Baseline
When using **raw** Deep Reinforcement Learning agent scores as clustering data, a **baseline normalization csv-File** can be used to normalize these scores before clustering. **For each evaluation type used** - in the Deep Reinforcement Learning agent scores - a baseline score- and random score-column needs to be provided in the baseline csv-File. Therefore the csv-File should have the following format:  
| Games | Baseline_1 | Random_1 | ... | Baseline_n | Random_n |  
| :---: | :---: | :---: | :---: | :---: | :---: |
| GameName_1 | w_1 | x_1 | ... | y_1 | z_1 |  
| ... | ... | ... | ... | ... | ... |   
| GameName_n| w_n | x_n | ... | y_n | z_n  

If the original data only contains **Deep Reinforcement Learning agent scores from a single type of evaluation method**, data can be simply normalized by calling *convertDRLScores(filename)* on the object created in the beginning. Otherwise, a list of start indexes and end indexes of the corresponding scores start- and end-columns need to be provided:  
```python
startIndexes = [0,8]
endIndexes = [9,15]

obj.convertDRLScores(filename, startIndexes, endIndexes)
```