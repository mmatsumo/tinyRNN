# cognitive_dynamics

Discovering Cognitive Strategies with Tiny Recurrent Neural Networks

## System requirements
- Tested on Windows and Linux
- Python 3
- Python packages: pytorch, scikit-learn, numpy, scipy, pandas=1.5.3, matplotlib, joblib=1.2.0, tqdm

## Installation guide
1. Clone the repository
2. Install the required packages (few minutes; recommended to use a virtual environment like conda)


[//]: # (Instructions to run on data)

[//]: # (Expected output)

[//]: # (Expected run time for demo)

## Instructions for use and Demo
- Start from main.py 
- In each experiment, the models are first trained (exp\*.py), analyzed (ana\*.py), then plotted (plotting\*.py).
- In the training_experiments, agents are trained on some datasets.
- In the analysis_experiments, the trained agents are analyzed using various metrics.
- In the plotting_experiments, the results are plotted.
- In the simulating_experiments, the trained agents are used to simulate the data.
- Warning: Training all the experiments will require a long time, leading to around 5 million model instances.
- Simpler demo scripts to start with:
  - those scripts ending with "_minimal" in training_experiments: python main.py -t exp_monkeyV_minimal
  - those scripts ending with "_minimal" in analyzing_experiments: python main.py -a ana_monkey_minimal
  - those scripts ending with "_minimal" in plotting_experiments: python main.py -p plotting_monkey_minimal
  
## Additional notes related to settings:
### User-specific path issues:

data_path.json, containing *absolute* paths to the data files, should be created in the root directory of the project.

For example, the file should look like this:

{"BartoloMonkey": "D:\\p7ft2bvphx-1"}

All code in the project will use (mostly) the *relative* pathstored in path_settings.py; 
thus any scripts called by the console or the main file should from utils import goto_root_dir.


[//]: # (### Multiprocessing issues:)

[//]: # (- Advanced system settings)

[//]: # (- Advanced tab)

[//]: # (- Performance - Settings button)

[//]: # (- Advanced tab - Change button)

[//]: # (- Uncheck the "Automatically... " checkbox)

[//]: # (- Select the System managed size option box.)

