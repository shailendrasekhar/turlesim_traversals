import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen
import math
import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path


def evaluate_quadratic_bezier(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def evaluate_cubic_bezier(p0, p1, p2, p3, t):
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


class CharTrajectoryNode(Node):
    def __init__(self):
        super().__init__('char_trajectory_node')
        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        self.pen_client = self.create_client(SetPen, '/turtle1/set_pen')

        self.pose = None
        self.waypoints = []  # (x, y, pen_down)
        self.current_goal_index = 0
        self.pen_down = True
        self.timer = self.create_timer(0.05, self.move_turtle)

        char = input("Enter a character: ").upper()
        self.waypoints = self.get_waypoints_from_char(char, scale=5.5, offset=(3.0, 3.0), resolution=0.7)
        if not self.waypoints:
            self.get_logger().warn(f"Could not generate waypoints for character '{char}'")

    def pose_callback(self, msg):
        self.pose = msg

    def get_waypoints_from_char(self, char, scale=5.5, offset=(3.0, 3.0), resolution=0.7):
        try:
            font = FontProperties(family="DejaVu Sans", weight="bold")
            text_path = TextPath((0, 0), char, size=1.0, prop=font)
            verts = text_path.vertices
            codes = text_path.codes

            # Normalize and scale
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
                    waypoints.append((last_point[0], last_point[1], False))  # pen up
                    i += 1
                elif code == Path.LINETO:
                    pt = verts[i]
                    if last_point is not None and np.linalg.norm(pt - last_point) > resolution:
                        waypoints.append((pt[0], pt[1], True))
                    last_point = pt
                    i += 1
                elif code == Path.CURVE3:
                    p0 = last_point
                    p1 = verts[i]
                    p2 = verts[i+1]
                    for t in np.linspace(0, 1, num=20):
                        p = evaluate_quadratic_bezier(p0, p1, p2, t)
                        waypoints.append((p[0], p[1], True))
                    last_point = p2
                    i += 2
                elif code == Path.CURVE4:
                    p0 = last_point
                    p1 = verts[i]
                    p2 = verts[i+1]
                    p3 = verts[i+2]
                    for t in np.linspace(0, 1, num=20):
                        p = evaluate_cubic_bezier(p0, p1, p2, p3, t)
                        waypoints.append((p[0], p[1], True))
                    last_point = p3
                    i += 3
                elif code == Path.CLOSEPOLY:
                    i += 1  # skip
                else:
                    self.get_logger().warn(f"Unhandled Path code: {code}")
                    i += 1

            return waypoints

        except Exception as e:
            self.get_logger().error(f"Error generating waypoints: {str(e)}")
            return []

    def set_pen(self, off: bool):
        if not self.pen_client.service_is_ready():
            return

        req = SetPen.Request()
        req.r = 255
        req.g = 255
        req.b = 255
        req.width = 2
        req.off = off
        self.pen_client.call_async(req)

    def move_turtle(self):
        if not self.pose or self.current_goal_index >= len(self.waypoints):
            return

        goal_x, goal_y, should_draw = self.waypoints[self.current_goal_index]
        dx = goal_x - self.pose.x
        dy = goal_y - self.pose.y
        distance = math.sqrt(dx**2 + dy**2)

        # Pen state management
        if should_draw != self.pen_down:
            self.set_pen(off=not should_draw)
            self.pen_down = should_draw

        if distance < 0.15:
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.waypoints):
                self.get_logger().info("✅ Finished tracing character.")
                self.timer.cancel()
                self.publisher.publish(Twist())
            return

        angle_to_goal = math.atan2(dy, dx)
        angle_diff = (angle_to_goal - self.pose.theta + math.pi) % (2 * math.pi) - math.pi

        vel = Twist()
        vel.angular.z = max(min(6.0 * angle_diff, 2.0), -2.0)

        if abs(angle_diff) > 0.6:
            vel.linear.x = 0.1  # very slow when turning
        elif abs(angle_diff) > 0.3:
            vel.linear.x = 0.3
        else:
            vel.linear.x = min(1.8, 2.5 * distance)  # faster when aligned

        self.publisher.publish(vel)

def main():
    rclpy.init()
    node = CharTrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
