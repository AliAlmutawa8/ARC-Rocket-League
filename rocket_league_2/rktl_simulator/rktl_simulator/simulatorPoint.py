import random
from math import degrees

import pygame
import pymunk
import pymunk.pygame_util
import rclpy
from rclpy.node import Node
from rktl_simulator.simulator import (BALL_POS, CAR_POS, FIELD_HEIGHT, FIELD_WIDTH,
                       GOAL_DEPTH, Ball, Car)

from rktl_interfaces.msg import CarAction, Field, Pose


class PointGame(Node):
    def __init__(
        self,
        carStartList:list[tuple[bool, float, float, float] | tuple[bool, float, float]] = CAR_POS,
        ballPosition:tuple[float, float] = BALL_POS
    ):
        """Constructor method

        :param carStartList: Positions of cars, defaults to CAR_POS
        :type carStartList: list[tuple[bool, float, float, float]  |  tuple[bool, float, float]], optional
        :param ballPosition: Position of the ball, defaults to BALL_POS
        :type ballPosition: tuple[float, float], optional
        """        """Constructor method"""
        
        super().__init__("point_game_node")
        pygame.init()
        self.screen = pygame.display.set_mode((FIELD_WIDTH + GOAL_DEPTH, FIELD_HEIGHT))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()

        self.leftscore = 0
        self.rightscore = 0
        self.ticks = 60
        self.ballPosition = ballPosition
        self.carStartList = carStartList
        self.cars = []
        self.inputs = []
        self.exit = False

        self.gameSpace = pymunk.Space()
        
        self.publisher_ = self.create_publisher(Field, "simTopic", 10)
        self.subscriber_ = self.create_subscription(CarAction, "aiTopic", self.runAStep, 10)
        
    def run(self):
        """Main logic function to keep track of gamestate. Steps 0.1 seconds with each SPACE key press.
        :param steps: Number of steps taken per 0.1 seconds when SPACE key is pressed
        :type steps: int
        """
        
        self.ball = Ball(self.ballPosition[0], self.ballPosition[1], self.gameSpace, pymunk.vec2d.Vec2d(10,0))
        self.cars.append(Car(True, 2 * (FIELD_WIDTH + GOAL_DEPTH) / 3, FIELD_HEIGHT / 2, self.gameSpace, 180))
        
    def runAStep(self, msg: CarAction):
        """The callback function for the subscriber

        :param msg: The message sent in
        :type msg: CarAction
        """
        
        self.get_logger().info("Recieved message, calculating...")
        if not 0 <= msg.id < len(self.cars):
            self.get_logger().warn(f"Ignoring action for unknown car id {msg.id}")
            return
        touching = []
        self.ball.body.each_arbiter(lambda arb: touching.append(self.cars[0].shape in arb.shapes))
        if any(touching):  # shapes_collide() asserts when the shapes don't overlap in pymunk 7
            self.randomizePositions()
            
        steps = 10
        for _ in range(steps):
            self.cars[msg.id].update([msg.throttle, msg.steer])
            pygame.display.update()
            self.gameSpace.step(0.1/steps)
        msgOut = Field()
        msgOut.ball_pose.id = -1
        (msgOut.ball_pose.x, msgOut.ball_pose.y) = self.ball.getPos()
        msgOut.ball_pose.angle_degrees = 0.0
        (msgOut.ball_pose.vx, msgOut.ball_pose.vy) = self.ball.getVelocity()
        msgOut.ball_pose.z = msgOut.ball_pose.vz = 0.0  # planar sim: no height (see car_env.py for 2.5D)
        for i, c in enumerate(self.cars):
            tempPose = Pose()
            tempPose.id = i
            (tempPose.x, tempPose.y) = c.getPos()
            tempPose.angle_degrees = c.getAngle()
            (tempPose.vx, tempPose.vy) = c.getVelocity()
            tempPose.angular_velocity_dps = degrees(c.body.angular_velocity)
            if c.team:
                msgOut.team1_poses.append(tempPose)
            else:
                msgOut.team2_poses.append(tempPose)
        self.publisher_.publish(msgOut)
        self.get_logger().info("Calculated, published message")
    
    def randomizePositions(self):
        """Sets random positions and velocities for the ball and all cars
        """
        
        self.ball.setPos((
            random.uniform(0, FIELD_WIDTH),
            random.uniform(0, FIELD_HEIGHT)
        ))
        self.ball.setVelocity((
            random.uniform(-5, 5),
            random.uniform(-5, 5)
        ))
        for i, _ in enumerate(self.cars):
            self.cars[i].setPos((
                random.uniform(0, FIELD_WIDTH),
                random.uniform(0, FIELD_HEIGHT)
            ))
            self.cars[i].setVel((
                random.uniform(-5, 5),
                random.uniform(-5, 5)
            ))
            self.cars[i].setAngle(random.uniform(0, 360))

def main():
    rclpy.init()
    game = PointGame(carStartList=[
        [True, (FIELD_WIDTH + GOAL_DEPTH) / 3,FIELD_HEIGHT / 2]
    ])
    game.run()
    rclpy.spin(game)
    game.destroy_node()
    rclpy.shutdown()