## LLM Based RobotArm Simulation
This project uses llm to control a 3DoF Robot Arm in a 3D simulation environment.

We created the 3D environment using OpenGL, the robot arm has RRT connect planner and lqr controller. To run this repo, you need to have your own GPT API key.

## Run Simulation
RRT only:
```bash
python llm_constraint.py
```
RRT + LQR:
```bash
python llm_constraint.py --enable-lqr
```
