Isaac Production

---

# Isaac Production
Isaac Production is a training platform for human-robot task allocation in manufacturing, built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) and [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html).

## System requirements
This project is developed and tested on Linux (Ubuntu 20.04) with Isaac Sim 4.5.0. It is expected to be compatible with newer Isaac Sim versions, but behavior may vary depending on driver and dependency differences.

For detailed simulator requirements (GPU, driver, CUDA, and OS support), please refer to the official [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).

## Setup Instructions

After installing Isaac Sim and Isaac Lab, and creating a compatible conda environment, complete the setup with the following steps:

1. In the `isaac-production` folder, create a symbolic link to your Isaac Sim installation:
`ln -s ${HOME}/isaacsim _isaac_sim`
2. Install the required Python package:
`pip install heapdict`

## Notes on released assets and code status
Some required simulation model source files (`.usd`) and offline path-planning route files (`.pkl`) are still private and are not included in this repository.

Although part of the codebase has been open-sourced on GitHub, this project is still being refined. Deployment instructions, code comments, and cleanup of unrelated code are not yet fully complete.


## Example: Running Training Jobs

To launch a batch of training jobs, run the following command from the project root directory:

```bash
bash batch_train.sh 1 3
```

This will execute the training scripts for test groups 1 through 3 as defined in `batch_train.sh`.

## Related Publications

**Journal of Manufacturing Systems**  
[Safe reinforcement learning with online filtering for fatigue-predictive human-robot task planning and allocation in production](https://doi.org/10.1016/j.jmsy.2025.12.019)  
Published in [Journal of Manufacturing Systems](https://www.sciencedirect.com/journal/journal-of-manufacturing-systems)  
DOI: [10.1016/j.jmsy.2025.12.019](https://doi.org/10.1016/j.jmsy.2025.12.019)

**Robotics and Computer-Integrated Manufacturing**  
[A hierarchical spatial-aware algorithm with efficient reinforcement learning for human-robot task planning and allocation in production](https://doi.org/10.1016/j.rcim.2025.103159)  
Published in [Robotics and Computer-Integrated Manufacturing](https://www.sciencedirect.com/journal/robotics-and-computer-integrated-manufacturing)  
DOI: [10.1016/j.rcim.2025.103159](https://doi.org/10.1016/j.rcim.2025.103159)
