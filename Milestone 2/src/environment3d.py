import numpy as np

class DynamicObstacle3D:
    def __init__(self, x, y, z, radius, vx=0.0, vy=0.0, vz=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.radius = float(radius)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
        yield self.radius

    def update(self, dt, width, height, depth):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        if self.x - self.radius <= 0:
            self.x, self.vx = self.radius, abs(self.vx)
        elif self.x + self.radius >= width:
            self.x, self.vx = width - self.radius, -abs(self.vx)

        if self.y - self.radius <= 0:
            self.y, self.vy = self.radius, abs(self.vy)
        elif self.y + self.radius >= height:
            self.y, self.vy = height - self.radius, -abs(self.vy)

        if self.z - self.radius <= 0:
            self.z, self.vz = self.radius, abs(self.vz)
        elif self.z + self.radius >= depth:
            self.z, self.vz = depth - self.radius, -abs(self.vz)

class Environment3D:
    def __init__(self):
        import config
        self.width = config.width
        self.height = config.height
        self.depth = config.width
        self.dt = config.dt
        self.boundary_margin = config.boundary_margin
        
        self.target_waypoint = None # Added for user interactivity
        
        self.obstacles = [
            (self.width*0.25, self.height*0.5, self.depth*0.5, 80.0),
            (self.width*0.75, self.height*0.5, self.depth*0.5, 80.0)
        ]
        
        self.dynamic_obstacles = [
            DynamicObstacle3D(self.width/2, self.height/2, self.depth/2, 60.0, vx=60, vy=50, vz=-40)
        ]

    @property
    def all_obstacles(self):
        return self.obstacles + [tuple(d) for d in self.dynamic_obstacles]

    def step(self, dt=None):
        if dt is None: 
            dt = self.dt
        for d in self.dynamic_obstacles:
            d.update(dt, self.width, self.height, self.depth)
