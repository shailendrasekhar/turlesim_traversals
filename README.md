# Navigational Experiments with Turtlesim

This ROS 2 package uses `turtlesim` to have multiple turtles collaboratively draw letters or strings on the screen.

## Prerequisites

- ROS 2 Humble (or a compatible ROS 2 distribution)
- `turtlesim` package (install with `sudo apt install ros-humble-turtlesim`)

## Installation & Build

1. Clone this repository into your ROS 2 workspace's `src/` folder:

    ```bash
    cd ~/ros2_ws/src
    git clone <your-repo-url>
    ```

2. Build the package and source the setup file:

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select turtle_commands
    source install/setup.bash
    ```

## Usage

Launch the `turtlesim_node` in one terminal:

```bash
ros2 run turtlesim turtlesim_node
```

In another terminal (with your workspace sourced), run the multi-turtle drawing node:

```bash
ros2 run turtle_commands char_trajectory
```

You will be prompted for:

1. **Number of turtles** to use 
2. **Character(s) to draw**:
   - A single uppercase letter, e.g. `A`.
   - The keyword `all` to draw the entire uppercase alphabet A–Z in sequence.

The script will:
- Clear the screen
- Reset all turtles back to the starting position (5.5, 5.5)
- Draw each character in turn with the selected turtles

Press `Ctrl+C` at any time to abort.

