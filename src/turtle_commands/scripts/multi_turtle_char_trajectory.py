import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen, TeleportAbsolute
import math
import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path


def evaluate_quadratic_bezier(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def evaluate_cubic_bezier(p0, p1, p2, p3, t):
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def split_waypoints_contiguous(waypoints, num_turtles):
    chunk_size = len(waypoints) // num_turtles
    chunks = [waypoints[i * chunk_size:(i + 1) * chunk_size] for i in range(num_turtles - 1)]
    chunks.append(waypoints[(num_turtles - 1) * chunk_size:])

    # 🛠 Force each turtle’s first point to pen-up
    for chunk in chunks:
        if chunk:
            x, y, _ = chunk[0]
            chunk[0] = (x, y, False)
    return chunks


class TurtleAgent:
    def __init__(self, node: Node, name: str, waypoints):
        self.node = node
        self.name = name
        self.pose = None
        self.waypoints = waypoints
        self.index = 0
        self.pen_down = True
        self.initialized = False

        self.pub = node.create_publisher(Twist, f'/{name}/cmd_vel', 10)
        self.sub = node.create_subscription(Pose, f'/{name}/pose', self.pose_callback, 10)
        self.pen_client = node.create_client(SetPen, f'/{name}/set_pen')
        self.teleport_client = node.create_client(TeleportAbsolute, f'/{name}/teleport_absolute')

        if waypoints:
            self.teleport_to_start()

    def pose_callback(self, msg):
        self.pose = msg

    def teleport_to_start(self):
        if not self.teleport_client.service_is_ready():
            return

        x, y, _ = self.waypoints[0]
        self.set_pen(off=True)

        req = TeleportAbsolute.Request()
        req.x = float(x)
        req.y = float(y)
        req.theta = 0.0
        self.teleport_client.call_async(req)

        self.node.get_logger().info(f"{self.name} teleported to start at ({x:.2f}, {y:.2f})")
        self.initialized = True

    def set_pen(self, off: bool):
        if not self.pen_client.service_is_ready():
            return
        req = SetPen.Request()
        req.r = np.random.randint(100, 255)
        req.g = np.random.randint(100, 255)
        req.b = np.random.randint(100, 255)
        req.width = 2
        req.off = off
        self.pen_client.call_async(req)

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

        if distance < 0.15:
            self.index += 1
            if self.index >= len(self.waypoints):
                self.pub.publish(Twist())
            return

        angle_to_goal = math.atan2(dy, dx)
        angle_diff = (angle_to_goal - self.pose.theta + math.pi) % (2 * math.pi) - math.pi

        vel = Twist()
        vel.angular.z = max(min(6.0 * angle_diff, 2.0), -2.0)

        if abs(angle_diff) > 0.6:
            vel.linear.x = 0.1
        elif abs(angle_diff) > 0.3:
            vel.linear.x = 0.3
        else:
            vel.linear.x = min(1.8, 2.5 * distance)

        self.pub.publish(vel)


class MultiTurtleCharDrawer(Node):
    def __init__(self):
        super().__init__('multi_turtle_char_drawer')
        char = input("Enter a character to draw: ").upper()
        num_turtles = int(input("How many turtles? (e.g., 2–4): "))

        waypoints = self.get_waypoints_from_char(char, scale=5.5, offset=(3.0, 3.0), resolution=0.7)
        if len(waypoints) < num_turtles:
            self.get_logger().warn("Not enough waypoints to split among turtles!")
            num_turtles = 1

        chunks = split_waypoints_contiguous(waypoints, num_turtles)
        turtle_names = [f"turtle{i+1}" for i in range(num_turtles)]

        self.turtles = []
        for name, chunk in zip(turtle_names, chunks):
            agent = TurtleAgent(self, name, chunk)
            self.turtles.append(agent)

        self.timer = self.create_timer(0.05, self.update_all)

    def update_all(self):
        for turtle in self.turtles:
            turtle.move()

    def get_waypoints_from_char(self, char, scale=5.5, offset=(3.0, 3.0), resolution=0.7):
        try:
            font = FontProperties(family="DejaVu Sans", weight="bold")
            path = TextPath((0, 0), char, size=1.0, prop=font)
            verts = path.vertices
            codes = path.codes

            x_min, y_min = np.min(verts, axis=0)
            x_max, y_max = np.max(verts, axis=0)
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
                    elif np.linalg.norm(pt - last_point) > resolution:
                        waypoints.append((pt[0], pt[1], True))
                    last_point = pt
                    i += 1
                elif code == Path.CURVE3:
                    p0 = last_point
                    p1 = verts[i]
                    p2 = verts[i + 1]
                    for t in np.linspace(0, 1, 20):
                        p = evaluate_quadratic_bezier(p0, p1, p2, t)
                        waypoints.append((p[0], p[1], True))
                    last_point = p2
                    i += 2
                elif code == Path.CURVE4:
                    p0 = last_point
                    p1 = verts[i]
                    p2 = verts[i + 1]
                    p3 = verts[i + 2]
                    for t in np.linspace(0, 1, 20):
                        p = evaluate_cubic_bezier(p0, p1, p2, p3, t)
                        waypoints.append((p[0], p[1], True))
                    last_point = p3
                    i += 3
                elif code == Path.CLOSEPOLY:
                    i += 1
                else:
                    i += 1
            return waypoints
        except Exception as e:
            self.get_logger().error(f"Error generating path: {str(e)}")
            return []


def main():
    rclpy.init()
    node = MultiTurtleCharDrawer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
