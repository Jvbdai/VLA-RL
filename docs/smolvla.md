# SmolVLA CloseFridge rollout

Use the `vla-rl` Conda environment (Python 3.12), not the original `robocasa`
environment. The new environment reuses the `robocasa/` and `robosuite/` clones
and downloaded assets. The project root is the parent of those clones.

From a Windows Command Prompt in the project root:

```bat
run_smolvla.cmd
```

This activates the new environment automatically, runs one pretrained SmolVLA
episode, and saves videos and evaluation metrics beneath `runs/`. It does not
open the keyboard teleoperation window, train a policy, or add residual RL.
Initial use downloads the pretrained model and tokenizer from Hugging Face.

```bat
run_smolvla.cmd --steps 20
run_smolvla.cmd --episodes 20
```

The first command is a short integration check, not a task-performance benchmark.
The second uses full task horizons. Use distinct held-out seeds and layouts for
research evaluation; this launcher alone does not implement the research protocol.

For PyCharm, add an existing local interpreter:
`C:\Users\Joshua\miniconda3\envs\vla-rl\python.exe`.
Run `scripts/run_smolvla.py` using that interpreter. The launcher handles the
working directory to avoid the outer RoboCasa clone shadowing the Python package.

Camera mapping follows the checkpoint training convention: left agent view to
camera1, right agent view to camera2, wrist view to camera3. The upstream wrapper
handles 16-dimensional proprioception and the 12-dimensional PandaOmron action.

Upstream integration:
https://huggingface.co/docs/lerobot/main/robocasa

LeRobot checkout at setup: `45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6`.
RoboCasa's declared `lerobot==0.3.3` dependency conflicts with this integration;
install its existing clone with `--no-deps` in this environment. Do not run its
unqualified `pip install -e .` here, which would restore the older LeRobot.
