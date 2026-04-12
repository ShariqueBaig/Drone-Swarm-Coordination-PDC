"""
comm_monitor.py - Monitor communication overhead between drones
Add hooks to swarm.py to track messages
"""

import csv
import time
import numpy as np  # ← ADD THIS IMPORT
from datetime import datetime

class CommunicationMonitor:
    def __init__(self, log_file="comm_overhead.csv"):
        self.log_file = log_file
        self.frame_count = 0
        self.total_messages = 0
        self.messages_per_frame = []
        
        # Create CSV
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame', 'total_messages', 
                'messages_per_second', 'avg_messages_per_drone'
            ])
    
    def start_frame(self):
        self.frame_start = time.time()
    
    def record_messages(self, num_messages, num_drones):
        """Record number of messages sent this frame"""
        self.total_messages += num_messages
        self.messages_per_frame.append(num_messages)
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            elapsed = time.time() - self.frame_start
            msgs_per_sec = num_messages / elapsed if elapsed > 0 else 0
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.frame_count,
                    self.total_messages,
                    f"{msgs_per_sec:.2f}",
                    f"{num_messages / num_drones:.2f}"
                ])
            
            print(f"[COMM] Frame {self.frame_count}: {num_messages} msgs | "
                  f"{msgs_per_sec:.1f} msg/s | {num_messages/num_drones:.2f} per drone")

def count_communication_messages(comm_mask, num_drones):
    """Count messages that would be sent in auction_tasks"""
    # Each pair of drones within comm radius potentially sends messages
    messages = np.sum(comm_mask) - num_drones  # Subtract self
    return max(0, messages)