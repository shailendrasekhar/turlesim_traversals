import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen, TeleportAbsolute, Spawn
from std_srvs.srv import Empty # Added for /clear service
import math
import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
import time # Added for sleep
import string # Added for alphabet


def evaluate_quadratic_bezier(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def evaluate_cubic_bezier(p0, p1, p2, p3, t):
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def split_waypoints_evenly(waypoints, num_turtles):
    """Distributes waypoints as evenly as possible among turtles."""
    n_points = len(waypoints)
    if n_points == 0 or num_turtles <= 0:
        return [[] for _ in range(num_turtles)] # Return empty lists if no points/turtles

    base_chunk_size = n_points // num_turtles
    remainder = n_points % num_turtles

    chunks = []
    start_index = 0
    for i in range(num_turtles):
        chunk_size = base_chunk_size + (1 if i < remainder else 0)
        end_index = start_index + chunk_size
        chunk = list(waypoints[start_index:end_index])
        chunks.append(chunk)
        start_index = end_index

    # --- Bridge the Gap Logic --- 
    for i in range(1, num_turtles): # Start from the second chunk
        if chunks[i-1] and chunks[i]: # Ensure previous and current chunks exist
            last_point_prev = chunks[i-1][-1]
            first_point_curr = chunks[i][0]
            # Compare coordinates (ignore pen state)
            if abs(last_point_prev[0] - first_point_curr[0]) > 1e-4 or \
               abs(last_point_prev[1] - first_point_curr[1]) > 1e-4:
                # Create a bridging point (pen up)
                bridge_point = (last_point_prev[0], last_point_prev[1], False)
                # Insert at the beginning of the current chunk
                chunks[i].insert(0, bridge_point)
                # print(f"Bridging gap between turtle {i} and {i+1}") # Optional debug log
    # --------------------------

    # Force first point pen-up (applies to original or bridge point)
    for chunk in chunks:
        if chunk:
            x, y, _ = chunk[0]
            chunk[0] = (x, y, False) 
    return chunks


def split_waypoints_by_strokes(waypoints, num_turtles):
    """Splits waypoints into chunks based on drawing strokes (pen up/down)."""
    if not waypoints or num_turtles <= 0:
        return [[] for _ in range(num_turtles)]

    strokes = []
    current_stroke = []
    for i, point in enumerate(waypoints):
        # Start a new stroke if pen is up (should_draw=False), unless it's the very first point
        if not point[2] and i > 0: 
            if current_stroke: # Add the completed stroke
                strokes.append(current_stroke)
            current_stroke = [point] # Start new stroke
        else:
            current_stroke.append(point)
    
    if current_stroke: # Add the last stroke
        strokes.append(current_stroke)

    # If no pen-up points were found (single stroke path), fall back to even splitting
    if len(strokes) <= 1:
        # print("Warning: Path contains only one stroke. Falling back to even point splitting with bridge.")
        return split_waypoints_evenly(waypoints, num_turtles) # Use the modified even split
    
    # Distribute strokes among turtles, aiming for even point count per turtle
    total_points = len(waypoints)
    # target_points_per_turtle = total_points / num_turtles # Not strictly used in greedy
    chunks = [[] for _ in range(num_turtles)]
    turtle_point_counts = [0] * num_turtles
    stroke_assignments = [[] for _ in range(num_turtles)] # Keep track of stroke indices

    # Assign strokes greedily to the turtle currently most below target point count
    for stroke_idx, stroke in enumerate(strokes):
        # Find turtle with fewest points currently assigned
        min_points = float('inf')
        target_turtle_idx = 0
        for i in range(num_turtles):
            if turtle_point_counts[i] < min_points:
                min_points = turtle_point_counts[i]
                target_turtle_idx = i
        
        stroke_assignments[target_turtle_idx].append(stroke_idx)
        turtle_point_counts[target_turtle_idx] += len(stroke)

    # Reconstruct waypoint chunks from assigned strokes
    temp_chunks = [[] for _ in range(num_turtles)]
    for i in range(num_turtles):
        assigned_strokes = [strokes[idx] for idx in stroke_assignments[i]]
        temp_chunks[i] = [point for stroke in assigned_strokes for point in stroke]

    # --- Bridge the Gap Logic (for stroke-based splits) --- 
    chunks = temp_chunks # Start with the stroke-based chunks
    # We need to compare the logical end/start even if split by strokes
    # This is harder because the assigned strokes might not be sequential in the original list
    # For now, let's apply the simpler coordinate check as in split_evenly, although less robust here.
    for i in range(1, num_turtles):
         if chunks[i-1] and chunks[i]:
            last_point_prev = chunks[i-1][-1]
            first_point_curr = chunks[i][0]
            if abs(last_point_prev[0] - first_point_curr[0]) > 1e-4 or \
               abs(last_point_prev[1] - first_point_curr[1]) > 1e-4:
                bridge_point = (last_point_prev[0], last_point_prev[1], False)
                chunks[i].insert(0, bridge_point)
                # print(f"Bridging stroke gap between turtle {i} and {i+1}") # Optional debug log
    # -----------------------------------------------------------

    # Ensure first point of each non-empty chunk has pen up
    for chunk in chunks:
        if chunk:
            x, y, _ = chunk[0]
            chunk[0] = (x, y, False)

    return chunks


class TurtleAgent:
    START_X, START_Y, START_THETA = 5.5, 5.5, 0.0 # Default start pose

    def __init__(self, node: Node, name: str): # Removed waypoints from init
        self.node = node
        self.name = name
        self.pose = None
        self.waypoints = [] # Initialize empty
        self.index = 0
        self.pen_down = False
        self.initialized = False # Will be set true after first reset/teleport

        self.pub = node.create_publisher(Twist, f'/{name}/cmd_vel', 10)
        self.sub = node.create_subscription(Pose, f'/{name}/pose', self.pose_callback, 10)
        self.pen_client = node.create_client(SetPen, f'/{name}/set_pen')
        self.teleport_client = node.create_client(TeleportAbsolute, f'/{name}/teleport_absolute')

        # Wait for services on first init is good practice
        while rclpy.ok() and not self.pen_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f'[{self.name}] set_pen service not available, waiting...')
        while rclpy.ok() and not self.teleport_client.wait_for_service(timeout_sec=1.0):
             self.node.get_logger().info(f'[{self.name}] teleport_absolute service not available, waiting...')


    def reset(self, new_waypoints):
        """Resets the turtle's state for a new path."""
        self.node.get_logger().info(f"[{self.name}] Resetting with {len(new_waypoints)} waypoints.")
        self.waypoints = new_waypoints
        self.index = 0
        self.pen_down = False
        self.initialized = True # Mark as ready to move after reset
        self.set_pen(off=True) # Ensure pen is up after reset

    def pose_callback(self, msg):
        self.pose = msg

    def teleport_absolute(self, x, y, theta): # Changed method signature
        if not self.teleport_client.service_is_ready():
            self.node.get_logger().warn(f'[{self.name}] Teleport service not ready.')
            return
        self.initialized = False # Temporarily disable movement during teleport
        self.set_pen(off=True) # Ensure pen is up before teleporting

        req = TeleportAbsolute.Request()
        req.x = float(x)
        req.y = float(y)
        req.theta = float(theta)
        
        # Use call, not call_async, to wait for teleport completion
        future = self.teleport_client.call_async(req)
        # We might not strictly need to wait here if the main loop pause is sufficient
        # rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0) # Optional wait

        self.node.get_logger().info(f"[{self.name}] Teleported to ({x:.2f}, {y:.2f}, {theta:.2f})")
        # Update internal pose estimate immediately after teleport request
        # A proper wait/callback on future would be more robust
        if self.pose is None: self.pose = Pose()
        self.pose.x = x
        self.pose.y = y
        self.pose.theta = theta
        self.initialized = True # Re-enable movement

    def set_pen(self, off: bool):
        if not self.pen_client.service_is_ready():
            self.node.get_logger().warn(f'[{self.name}] SetPen service not ready.')
            return
        req = SetPen.Request()
        # Keep pen color consistent for now, maybe randomize per character later
        req.r = 200 
        req.g = 200
        req.b = 200
        req.width = 2
        req.off = off
        # Use call_async as we don't strictly need to wait for pen change confirmation
        self.pen_client.call_async(req)

    def is_finished(self):
        """Checks if the turtle has completed its current path."""
        return self.initialized and self.index >= len(self.waypoints)

    def move(self):
        if not self.pose or not self.initialized or self.index >= len(self.waypoints):
            return

        goal_x, goal_y, should_draw = self.waypoints[self.index]
        dx = goal_x - self.pose.x
        dy = goal_y - self.pose.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if should_draw != self.pen_down:
            self.set_pen(off=not should_draw)
            self.pen_down = should_draw 

        # Decrease distance threshold for reaching waypoint
        if distance < 0.1:
            self.index += 1
            if self.index >= len(self.waypoints):
                self.node.get_logger().info(f"[{self.name}] Task complete.")
                self.pub.publish(Twist()) 
            return 
        
        # --- Movement Calculation ---
        angle_to_goal = math.atan2(dy, dx)
        angle_diff = (angle_to_goal - self.pose.theta + math.pi) % (2 * math.pi) - math.pi

        vel = Twist()
        vel.angular.z = max(min(6.0 * angle_diff, 3.0), -3.0) 
        if abs(angle_diff) > 0.6:
            vel.linear.x = 0.2 
        elif abs(angle_diff) > 0.3:
            vel.linear.x = 0.5 
        else:
            vel.linear.x = min(2.0, 3.0 * distance) 

        self.pub.publish(vel)


class MultiTurtleCharDrawer(Node):
    def __init__(self, num_turtles: int, characters_to_process: list[str]):
        node_name = f'multi_turtle_char_drawer_{num_turtles}_turtles_{int(time.time())}'
        super().__init__(node_name) 
        
        self.num_turtles = num_turtles
        self.turtle_names = [f"turtle{i+1}" for i in range(self.num_turtles)]
        self.characters_to_draw = characters_to_process 
        self.current_char_index = 0
        self.turtles: list[TurtleAgent] = []
        self.is_resetting = False

        # --- Service Clients --- 
        self.clear_client = self.create_client(Empty, '/clear')
        self.spawn_client = self.create_client(Spawn, '/spawn') # Spawn client
        
        # Wait for essential services
        while rclpy.ok() and not self.clear_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/clear service not available, waiting...')
        while rclpy.ok() and not self.spawn_client.wait_for_service(timeout_sec=1.0):
             self.get_logger().info('/spawn service not available, waiting...')
        if not rclpy.ok(): raise RuntimeError("rclpy shut down during service wait")

        # --- Spawn Turtles --- 
        # Spawn turtles 2 through num_turtles, assuming turtle1 exists
        self.get_logger().info(f"Spawning turtles 2 through {self.num_turtles} (assuming turtle1 exists)...")
        spawn_futures = []
        # Modify loop range to start from 1 (for turtle2)
        for i in range(1, self.num_turtles):
            if not rclpy.ok(): break 
            name = self.turtle_names[i] # turtle_names list index matches i correctly here
            req = Spawn.Request()
            req.x = TurtleAgent.START_X
            req.y = TurtleAgent.START_Y
            req.theta = TurtleAgent.START_THETA
            req.name = name
            future = self.spawn_client.call_async(req)
            spawn_futures.append((name, future))

        # Wait for all spawns to complete
        for name, future in spawn_futures:
            if not rclpy.ok(): break 
            try:
                # Spin until the future is done, with a timeout
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if rclpy.ok():
                    if future.done() and future.result() is not None:
                         self.get_logger().info(f"Successfully spawned/verified '{future.result().name}'")
                    elif future.done():
                         self.get_logger().error(f"Failed to spawn '{name}': {future.exception()}")
                         # Consider exiting or handling this error more robustly
                    else:
                        self.get_logger().warn(f"Spawn service call for '{name}' timed out or returned no result.")
            except Exception as e:
                 if rclpy.ok(): self.get_logger().error(f"Exception while waiting for spawn of '{name}': {e}")
        if not rclpy.ok(): raise RuntimeError("rclpy shut down during spawn")

        # --- Initial Setup --- 
        if not self.characters_to_draw: # Handle empty list case
            self.get_logger().error("No characters provided to draw. Shutting down.")
            if rclpy.ok(): rclpy.shutdown()
            return
            
        self.get_logger().info(f"Initializing run for {self.num_turtles} turtles.")
        first_char = self.characters_to_draw[self.current_char_index]
        self.setup_turtles_for_char(first_char)
        
        if rclpy.ok():
            self.timer = self.create_timer(0.05, self.update_all)
            self.get_logger().info(f"Started drawing with {self.num_turtles} turtles. First char: '{first_char}'")
        else:
             raise RuntimeError("rclpy shut down before timer creation")


    def setup_turtles_for_char(self, char):
        """Generates waypoints and assigns them to turtles (creating or resetting)."""
        self.is_resetting = True # Pause updates
        self.get_logger().info(f"--- Setting up for character: '{char}' ---")
        
        waypoints = self.get_waypoints_from_char(char, scale=5.5, offset=(3.0, 3.0), resolution=0.7)
        if not waypoints:
             self.get_logger().error(f"Could not generate waypoints for '{char}'. Skipping.")
             # If waypoints fail, immediately try to trigger next character setup in update_all
             self.is_resetting = False
             # Ensure all turtles are marked finished so update_all proceeds
             for t in self.turtles: 
                 t.index = float('inf') # Mark as finished
             return 

        # Use stroke splitting (falls back to even splitting if needed)
        self.get_logger().info("Splitting waypoints by strokes...") # Keep this log
        chunks = split_waypoints_by_strokes(waypoints, self.num_turtles)
        
        # *** Add detailed logging for assigned waypoints ***
        for i, chunk in enumerate(chunks):
            self.get_logger().info(f"  Turtle {i+1} assigned {len(chunk)} waypoints:")
            # Log first few and last few points for brevity, or all if short
            if len(chunk) < 10:
                for pt_idx, pt in enumerate(chunk):
                    self.get_logger().info(f"    [{pt_idx}] ({pt[0]:.2f}, {pt[1]:.2f}, Draw={pt[2]})")
            else:
                for pt_idx in list(range(3)) + list(range(len(chunk)-3, len(chunk))):
                    pt = chunk[pt_idx]
                    indicator = "..." if pt_idx == 2 else f"[{pt_idx}]"
                    self.get_logger().info(f"    {indicator} ({pt[0]:.2f}, {pt[1]:.2f}, Draw={pt[2]})")
        # ****************************************************

        # Create or reset turtle agents
        if not self.turtles: # First time setup
             for i in range(self.num_turtles):
                agent = TurtleAgent(self, self.turtle_names[i])
                # Initial teleport is needed only on first creation
                agent.teleport_absolute(TurtleAgent.START_X, TurtleAgent.START_Y, TurtleAgent.START_THETA) 
                agent.reset(chunks[i])
                self.turtles.append(agent)
        else: # Reset existing turtles
             for i in range(self.num_turtles):
                if i < len(self.turtles):
                     self.turtles[i].reset(chunks[i])
                else:
                    # Handle case where num_turtles changed? (Not applicable here)
                    pass
        
        self.is_resetting = False # Resume updates


    def reset_environment_and_setup_next(self):
        """Clears screen, resets turtle positions, and sets up the next character."""
        self.is_resetting = True # Prevent move calls during reset
        self.get_logger().info("--- Resetting environment for next character ---")

        # 1. Clear screen
        if self.clear_client.service_is_ready():
            self.clear_client.call_async(Empty.Request())
            # No need to wait, can happen concurrently with teleport
        else:
             self.get_logger().warn("/clear service not ready during reset.")

        # 2. Teleport all turtles back to start
        for agent in self.turtles:
            # Check rclpy.ok() before calling service on agent's node
            if rclpy.ok():
                agent.teleport_absolute(TurtleAgent.START_X, TurtleAgent.START_Y, TurtleAgent.START_THETA)
            else:
                self.get_logger().warn("rclpy shutdown during teleport reset. Breaking loop.")
                break # Exit the loop if shutdown happens mid-teleport
        
        # Only proceed if rclpy is still ok
        if not rclpy.ok():
            self.is_resetting = False # Ensure flag is cleared
            return

        # 3. Brief pause to allow services to complete and for visual separation
        time.sleep(1.5) 

        # 4. Setup for the next character
        self.current_char_index += 1
        if self.current_char_index < len(self.characters_to_draw):
            next_char = self.characters_to_draw[self.current_char_index]
            self.setup_turtles_for_char(next_char) # This will set is_resetting=False
        else:
            # *** CHANGE: Initiate rclpy shutdown on completion ***
            self.get_logger().info(f"--- All characters drawn for {self.num_turtles} turtles! Initiating shutdown. ---")
            if self.timer: # Check if timer exists before cancelling
                 self.timer.cancel()
            # Setting is_resetting = False allows spin() to potentially exit if needed
            # but the cancelled timer is the primary mechanism.
            self.is_resetting = False 
            # Request shutdown. Spin in main will exit.
            if rclpy.ok():
                rclpy.shutdown()

        # This check might be redundant now as the completion logic is handled above
        # if self.current_char_index >= len(self.characters_to_draw):
        #      self.is_resetting = False


    def update_all(self):
        """Main loop: moves turtles and checks for character completion."""
        if self.is_resetting or not rclpy.ok(): # Also check rclpy status here
             return # Skip movement logic during reset sequence

        all_finished = True
        for turtle in self.turtles:
            if not turtle.is_finished():
                turtle.move()
                all_finished = False
            # Handle case where setup failed and turtle was marked finished
            elif turtle.index == float('inf'): 
                all_finished = True # Ensure this turtle is counted as finished to move on
        
        if all_finished:
            # Check if the timer is already cancelled (meaning the sequence finished)
            if self.timer and not self.timer.is_canceled():
                 # Check rclpy status before resetting
                 if rclpy.ok():
                    self.reset_environment_and_setup_next()
                 else:
                     self.get_logger().warn("rclpy shutdown detected in update_all before reset.")
                     if self.timer: self.timer.cancel() # Ensure timer stops


    def get_waypoints_from_char(self, char, scale=5.5, offset=(3.0, 3.0), resolution=0.7):
        try:
            font = FontProperties(family="DejaVu Sans", weight="bold")
            path = TextPath((0, 0), char, size=1.0, prop=font)
            verts = path.vertices
            codes = path.codes

            # Scale and offset logic...
            x_min, y_min = np.min(verts, axis=0)
            x_max, y_max = np.max(verts, axis=0)
            # Avoid division by zero for single-point paths (like '.') if used later
            if x_max == x_min: x_max += 1e-6 
            if y_max == y_min: y_max += 1e-6
            verts = (verts - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])
            verts *= scale
            verts += offset

            waypoints = []
            i = 0
            last_point = None

            while i < len(codes):
                code = codes[i]
                if code == Path.MOVETO:
                    last_point = verts[i]
                    waypoints.append((last_point[0], last_point[1], False))
                    i += 1
                elif code == Path.LINETO:
                    pt = verts[i]
                    if last_point is None:
                        waypoints.append((pt[0], pt[1], False))
                    else:
                        if np.linalg.norm(pt - last_point) > resolution or i == len(verts) -1:
                            waypoints.append((pt[0], pt[1], True))
                    last_point = pt
                    i += 1
                elif code == Path.CURVE3:
                    p0 = last_point if last_point is not None else verts[i]
                    p1 = verts[i]
                    p2 = verts[i + 1]
                    for t in np.linspace(0, 1, 10)[1:-1]:
                        p = evaluate_quadratic_bezier(p0, p1, p2, t)
                        waypoints.append((p[0], p[1], True))
                    waypoints.append((p2[0], p2[1], True))
                    last_point = p2
                    i += 2
                elif code == Path.CURVE4:
                    p0 = last_point if last_point is not None else verts[i]
                    p1 = verts[i]
                    p2 = verts[i + 1]
                    p3 = verts[i + 2]
                    for t in np.linspace(0, 1, 15)[1:-1]:
                        p = evaluate_cubic_bezier(p0, p1, p2, p3, t)
                        waypoints.append((p[0], p[1], True))
                    waypoints.append((p3[0], p3[1], True))
                    last_point = p3
                    i += 3
                elif code == Path.CLOSEPOLY:
                    # Optional: Add line back to start of *current* subpath if needed
                    # Requires tracking subpath start point.
                    # For most letters, this isn't strictly necessary visually.
                    i += 1 
                else:
                    i += 1 # Ignore unknown codes
            return waypoints
        except Exception as e:
            self.get_logger().error(f"Error generating path for '{char}': {str(e)}")
            return []


def main():
    rclpy.init()
    node = None # Initialize node to None
    num_turtles_to_run = 0
    chars_to_process = []

    # Get number of turtles
    while num_turtles_to_run <= 0 and rclpy.ok():
        try:
            num_turtles_to_run = int(input("Enter number of turtles to use (e.g., 1-4): "))
            if num_turtles_to_run <= 0:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError: # Handle Ctrl+D
            print("\nInput cancelled.")
            if rclpy.ok(): rclpy.shutdown()
            return
        if not rclpy.ok(): 
            print("rclpy shutdown detected during input. Exiting.")
            return 

    # Get character(s)
    while not chars_to_process and rclpy.ok():
        try:
            user_input = input("Enter character(s) to draw (e.g., 'A', 'B', 'C' or 'all'): ").strip()
            if not user_input: continue # Ask again if empty

            if user_input.lower() == 'all':
                chars_to_process = list(string.ascii_uppercase)
            elif len(user_input) == 1 and user_input.isalpha() and user_input.isupper():
                chars_to_process = [user_input]
            else:
                # Simple check for multiple chars separated by comma/space
                processed_input = []
                potential_chars = user_input.replace(',', ' ').split()
                valid = True
                for char in potential_chars:
                    if len(char) == 1 and char.isalpha() and char.isupper():
                        processed_input.append(char)
                    else:
                        valid = False
                        break
                if valid and processed_input:
                    chars_to_process = processed_input
                else:
                    print("Invalid input. Please enter one or more uppercase letters (A-Z), or 'all'.")
                    # Stay in the loop to ask again

        except EOFError: # Handle Ctrl+D
             print("\nInput cancelled.")
             if rclpy.ok(): rclpy.shutdown()
             return
        if not rclpy.ok(): 
            print("rclpy shutdown detected during input. Exiting.")
            return

    # Proceed only if we have characters and rclpy is ok
    if not chars_to_process or not rclpy.ok():
        print("No valid characters to draw or rclpy shut down. Exiting.")
        if rclpy.ok(): rclpy.shutdown()
        return

    try:
        print(f"\n===== Starting sequence for characters: {','.join(chars_to_process)} with {num_turtles_to_run} turtle(s) =====")
        # Pass the required number of turtles and character list
        node = MultiTurtleCharDrawer(num_turtles=num_turtles_to_run, characters_to_process=chars_to_process)
        if rclpy.ok(): 
            rclpy.spin(node) # Spin will return when the node calls rclpy.shutdown()
        
        # Check rclpy status after spin returns
        if rclpy.ok(): 
             print(f"===== Finished sequence ====")
        else:
             print("===== Sequence interrupted by shutdown ====")

    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt detected. Shutting down.")
    except RuntimeError as e:
         print(f"Runtime error during node execution: {e}")
    except Exception as e:
        print(f"\nAn unexpected exception occurred: {e}")
    finally:
        if node:
            print(f"Destroying node...")
            if rclpy.ok(): 
                node.destroy_node()
            else:
                print("(rclpy already shut down, cannot destroy node explicitly)")
        
        print("Shutting down rclpy context.")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
