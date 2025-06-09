#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
from dv import AedatFile  # 处理 .aedat4 文件
import numpy as np

parser = argparse.ArgumentParser(description="Visualize events from an AEDAT4 file.")
# parser.add_argument("--file", type=str, default="event.aedat4", required=True, help="Path to the .aedat4 file")
parser.add_argument("--buffer", type=int, default=2000, help="Number of events per visualization batch")
opt = parser.parse_args()

def visualize_events(aedat_path, buffer_size):
    with AedatFile(aedat_path) as f:
        # if not f.has_events:
        #     print("No event stream found in this file.")
        #     return

        events = f['events']  # get event stream

        # Visualization setup
        col_buffer, row_buffer, color_buffer = [], [], []
        color = ['red', 'green']
        s = plt.plot([], [])[0]
        plt.title("Event Visualization (.aedat4)")
        plt.xlabel("Column")
        plt.ylabel("Row")

        for event in events:
            col_buffer.append(event.x)
            row_buffer.append(event.y)
            color_buffer.append(color[event.polarity])

            if len(col_buffer) >= buffer_size:
                s.remove()
                s = plt.scatter(col_buffer, row_buffer, color=color_buffer, s=1)
                plt.pause(0.00001)
                col_buffer.clear()
                row_buffer.clear()
                color_buffer.clear()

        plt.show()

if __name__ == '__main__':
    file_path = "events20.aedat4"  # Default file path
    visualize_events(file_path, opt.buffer)
