# **Crowd Navigation Policies**

This project provides several **crowd navigation policies** that can be tested using Python scripts. Below, you’ll find details on how to **run the policies**, **configure the environment**, and **visualize predictions**.

## **Running Navigation Policies**

To run a navigation policy, use the following command:

```bash
python Test.py --policy POLICY
```
Available Policies:

- **campc**: Implements the SICNav policy.
- **dwa**: Uses the Dynamic Window Approach (DWA) for navigation.
- **orca_plus**: Uses the ORCA policy for crowd navigation.
- **ORCAPlusAll**: This policy creates an output.txt file containing ORCA-based future predictions for a few time steps ahead. However, these predictions are not used for navigation; only ORCA-based navigation is applied instead.
- **NewMPCChanging**: A policy under development that uses ORCAPlusAll for future predictions and controls the robot using Model Predictive Control (MPC).
- **NewMPC**: Another development policy that also uses ORCAPlusAll for future predictions and controls the robot using MPC.

## **Configuration**

The environment can be configured using the `env.config` file, where you can specify:

- **Number of humans**: Number of human agents present in the simulation.
- **Environment size**: Dimensions of the navigation environment.
- **Human radii**: Radii of human agents for collision avoidance calculations.

Changing the Environment Type

You can switch between different environment types in Test.py. Supported environment types include:

    hallway
    hallway_bottleneck


## **Visualizing ORCA Predictions**

To verify ORCA predictions, follow these steps:
## **1. Generate ORCA Predictions:**

Run this command to create the output.txt file containing future predictions using the ORCAPlusAll policy:

```bash

python Test.py --policy ORCAPlusAll
```
Note: Ensure the output.txt file is cleared or deleted before running this command to avoid overwriting issues.

## **2. Plot and Save the Predictions:**

Run Plot.py to visualize the predictions and save a video in the predictions folder:

```bash

python Plot.py
```
Important: Update the values in Plot.py (such as the number of humans, time horizon, plot size) according to the env.config file manually.







By following these steps, you can test different navigation policies, configure the environment, and visualize ORCA-based predictions.
