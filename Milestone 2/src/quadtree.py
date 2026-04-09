import numpy as np

class QuadTree:
    """Simple quadtree implementation for spatial partitioning"""
    
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary  # (x, y, width, height)
        self.capacity = capacity
        self.drones = []  # (id, position)
        self.divided = False
        self.children = []
        
    def insert(self, drone_id, position):
        """Insert a drone into the quadtree"""
        x, y = position
        bx, by, bw, bh = self.boundary
        
        # Check if point is in boundary
        if not (bx <= x <= bx + bw and by <= y <= by + bh):
            return False
            
        if len(self.drones) < self.capacity and not self.divided:
            self.drones.append((drone_id, position))
            return True
        else:
            if not self.divided:
                self._subdivide()
            
            for child in self.children:
                if child.insert(drone_id, position):
                    return True
        return False
    
    def _subdivide(self):
        """Split into 4 children"""
        x, y, w, h = self.boundary
        half_w, half_h = w/2, h/2
        
        self.children = [
            QuadTree((x, y, half_w, half_h), self.capacity),
            QuadTree((x + half_w, y, half_w, half_h), self.capacity),
            QuadTree((x, y + half_h, half_w, half_h), self.capacity),
            QuadTree((x + half_w, y + half_h, half_w, half_h), self.capacity)
        ]
        
        # Redistribute existing drones
        for drone in self.drones:
            for child in self.children:
                if child.insert(drone[0], drone[1]):
                    break
        self.drones = []
        self.divided = True
        
    def query_radius(self, position, radius):
        """Find all drones within radius of position"""
        x, y = position
        bx, by, bw, bh = self.boundary
        
        # Check if search circle intersects boundary
        if not self._circle_intersects_boundary(x, y, radius):
            return []
            
        results = []
        
        # Check drones in this node
        for drone_id, drone_pos in self.drones:
            if np.linalg.norm(drone_pos - position) <= radius:
                results.append((drone_id, drone_pos))
                
        # Check children
        if self.divided:
            for child in self.children:
                results.extend(child.query_radius(position, radius))
                
        return results
    
    def _circle_intersects_boundary(self, cx, cy, r):
        """Check if circle intersects with boundary"""
        bx, by, bw, bh = self.boundary
        
        # Find closest point on rectangle to circle
        closest_x = max(bx, min(cx, bx + bw))
        closest_y = max(by, min(cy, by + bh))
        
        # Distance from circle center to closest point
        distance = np.sqrt((cx - closest_x)**2 + (cy - closest_y)**2)
        
        return distance <= r