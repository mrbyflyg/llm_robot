## LLM Based RobotArm Simulation
This project uses llm to control a 3DoF Robot Arm in a 3D simulation environment.

We created the 3D environment using OpenGL, the robot arm has RRT connect planner and lqr controller. To run this repo, you need to have your own GPT API key.

Sample user command1: move the apple/banana to the target position/bowl/box.

Sample user command2: move the apple along z axis for 0.2m.

## Run Simulation
RRT only:
```bash
python llm_constraint.py
```
RRT + LQR:
```bash
python llm_constraint.py --enable-lqr
```



https://github.com/user-attachments/assets/6044b526-478d-49c8-a2fe-0ab0e6183f8d


