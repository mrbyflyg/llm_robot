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


\documentclass{article}
\usepackage{media9}
\begin{document}
\includemedia[
  width=0.4\linewidth,
  height=0.3\linewidth,
  activate=pageopen,
  addresource=demo.mp4,
  flashvars={source=videos/Ruinn.MP4}]{}{VPlayer.swf}
 \end{document}
