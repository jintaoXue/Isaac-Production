Isaac Production

---

# Isaac Production
Isaac Production is a training platform for human-robot task allocation in manufacturing, built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) and [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html).

## Setup after deploying Isaac Lab and creating a conda environment
In the `isaac-production` folder, link the Isaac Sim repository:
`ln -s ${HOME}/isaacsim _isaac_sim`

Then install the required package:
`pip install heapdict`

## Notes on released assets and code status
Some required simulation model source files (`.usd`) and offline path-planning route files (`.pkl`) are still private and are not included in this repository.

Although part of the codebase has been open-sourced on GitHub, this project is still being refined. Deployment instructions, code comments, and cleanup of unrelated code are not yet fully complete.


## Run order
`bash batch_train.sh 1 3`

## Related publications

Published in *Journal of Manufacturing Systems*:
*Safe reinforcement learning with online filtering for fatigue-predictive human-robot task planning and allocation in production*.

Published in *Robotics and Computer-Integrated Manufacturing*:
*A hierarchical spatial-aware algorithm with efficient reinforcement learning for human-robot task planning and allocation in production*.


