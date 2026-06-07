import tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import sys
import threading
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


# 全局变量
is_confirmed = False
received_grid = None
grid_publisher = None
node = None

class GridSubscriber(Node):
    def __init__(self):
        super().__init__("grid_subscriber")
        custom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.subscription = self.create_subscription(
            Int32,
            '/encoded_grid',
            self.listen_callback,
            custom_qos
        )
    def listen_callback(self, msg):
        global received_grid
        received_grid = msg.data
        print(f">>> 收到消息：{msg.data}") 

def ros_spin(node):
    rclpy.spin(node)

def encode_grid(all_cell_list):
    encoded_value = 0
    bit_position = 0
    for i in range(3, -1, -1):
        for j in range(3):
            target_row = 4 - i
            target_col = j + 1
            cell_value = 0
            for cell in all_cell_list:
                if cell.row == target_row and cell.col == target_col:
                    cell_value = int(cell.text) if cell.text else 0
                    break
            encoded_value |= (cell_value << bit_position)
            bit_position += 2
    return encoded_value 

class OneCell:
    TOTAL_REL_WIDTH = 0.22
    CELL_REL_HEIGHT = 0.15
    GAP_AS_FRACTION_OF_BUTTON_WIDTH = 0.1

    def __init__(self, root, rel_x, rel_y, row, col):
        self.root = root
        self.buttons = []
        self.text = None
        self.rel_x_start = rel_x
        self.rel_y = rel_y
        self.row = row
        self.col = col
        self.is_locked = False

        if row == 1:
            button_texts = ["1", "2", "0"]
        elif row in [2, 3] and col == 2:
            button_texts = ["2", "3", "0"]
        else:
            button_texts = ["1", "2", "3", "0"]
        
        num_buttons = len(button_texts)

        button_width = self.TOTAL_REL_WIDTH / (num_buttons + (num_buttons - 1) * self.GAP_AS_FRACTION_OF_BUTTON_WIDTH)
        gap_width = self.GAP_AS_FRACTION_OF_BUTTON_WIDTH * button_width

        for i, btn_text in enumerate(button_texts):
            btn_relx = self.rel_x_start + i * (button_width + gap_width)

            btn = tk.Button(
                self.root,
                text=btn_text,
                command=lambda t=btn_text: self._action(t),
                bg='white',
                fg='black',
            )
            btn.place(
                relx=btn_relx,
                rely=self.rel_y,
                relwidth=button_width,
                relheight=self.CELL_REL_HEIGHT,
            )
            
            self.buttons.append(btn)

    def _action(self, button_text):
        if self.is_locked:
            return
        color_map = {
            "1": "green",
            "2": "blue",
            "3": "red",
        }
        self.text = button_text
        if button_text != "0":
            new_color = color_map.get(button_text, "white")
            for btn in self.buttons:
                btn.config(bg=new_color, fg="white")
        else:
            for btn in self.buttons:
                btn.config(bg="white", fg="black")

    def lock(self):
        self.is_locked = True

    def reset(self):
        if self.is_locked:
            return
        self._action("0")

    def is_active(self):
        return self.text != "0"

    def get_info(self):
        return {
            "row": self.row,
            "col": self.col,
            "num": self.text,
        }
    
    def set_from_code(self, code):
        if self.is_locked:
            return
        text = {0: "0", 1: "1", 2: "2", 3: "3"}
        if code in text:
            self._action(text[code])


# 清除所有单元格
def clear_all_cells(all_cells_list):
    global is_confirmed
    for cell in all_cells_list:
        cell.reset()

# 确认所有单元格
def confirm_all_cells(all_cells_list):
    global is_confirmed
    if is_confirmed:
        return
    
    final_value = encode_grid(all_cells_list)
    print(f"确认并发送最终值: {final_value}")
    
    msg = Int32()
    msg.data = final_value
    grid_publisher.publish(msg)
    
    is_confirmed = True
    for cell in all_cells_list:
        cell.lock()

def decode_grid(encoded_value):
    if encoded_value == 0:
        return [[0 for _ in range(3)] for _ in range(4)]
    grid = [[0 for _ in range(3)] for _ in range(4)]
    bit_position = 0
    for i in range(3, -1, -1): 
        for j in range(3): 
            value = (encoded_value >> bit_position) & 0x3
            value = max(0, min(3, value))
            grid[i][j] = value
            bit_position += 2
    return grid

def check_ros_message(all_cells_list):
    global received_grid
    try:
        if received_grid is not None:
            grid_data = decode_grid(received_grid)
            print(grid_data)
            for cell in all_cells_list:
                i = 4 - cell.row
                j = cell.col - 1
                if 0 <= i < 4 and 0 <= j < 3:
                    cell.set_from_code(grid_data[i][j])
            received_grid = None
    except Exception as e:
        print(f"Error: {e}")

    root.after(100, check_ros_message, all_cells_list)

if __name__ == "__main__":
    rclpy.init(args=None)
    node = GridSubscriber()
    grid_publisher = node.create_publisher(Int32, '/encoded_grid_python', 10)
    ros_thread = threading.Thread(target=ros_spin, args=(node,))
    ros_thread.daemon = True
    ros_thread.start()

    root = tk.Tk()
    root.title("测试")
    root.geometry("800x600")

    all_cells = []

    for i in range(4):
        for j in range(3):
            rel_x = 0.1 + j * 0.3
            rel_y = 0.2 + i * 0.2
            cell = OneCell(root, rel_x=rel_x, rel_y=rel_y, row=4-i, col=j+1)
            all_cells.append(cell)

    button_clear = tk.Button(
        root,
        text="清除",
        command=lambda: clear_all_cells(all_cells),
        bg='white',
        fg='black',
    )
    button_clear.place(
        relx=0.1,
        rely=0.05,
        relwidth=0.22,
        relheight=0.1,
    )

    button_sure = tk.Button(
        root,
        text="确认",
        command=lambda: confirm_all_cells(all_cells),
        bg='white',
        fg='black',
    )
    button_sure.place(
        relx=0.7,
        rely=0.05,
        relwidth=0.22,
        relheight=0.1,
    )

    root.after(1, check_ros_message, all_cells)
    print("等待ROS消息...")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
