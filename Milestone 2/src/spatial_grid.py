import numpy as np
from collections import defaultdict

class SpatialGrid:
    """Grid-based spatial partitioning for O(N) neighbor查找"""
    
    def __init__(self, cell_size, width, height):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.grid = defaultdict(list)
        
    def clear(self):
        """Clear the grid for next timestep"""
        self.grid.clear()
        
    def get_cell_coords(self, x, y):
        """Convert position to grid cell coordinates"""
        cell_x = int(x // self.cell_size)
        cell_y = int(y // self.cell_size)
        return (cell_x, cell_y)
    
    def insert_drones(self, positions, ids):
        """Insert all drones into grid"""
        for i, pos in enumerate(positions):
            cell = self.get_cell_coords(pos[0], pos[1])
            self.grid[cell].append((i, pos))
            
    def get_neighbors(self, pos, radius, include_self=False):
        """Get all drones within radius of pos using grid"""
        center_cell = self.get_cell_coords(pos[0], pos[1])
        radius_cells = int(np.ceil(radius / self.cell_size))
        
        neighbors = []
        
        # Check all cells within radius_cells
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                for drone_idx, drone_pos in self.grid.get(cell, []):
                    if not include_self and np.array_equal(drone_pos, pos):
                        continue
                    dist = np.linalg.norm(drone_pos - pos)
                    if dist <= radius:
                        neighbors.append((drone_idx, dist))
                        
        return neighbors