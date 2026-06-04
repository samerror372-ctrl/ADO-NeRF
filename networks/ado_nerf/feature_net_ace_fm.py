import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QGridLayout, 
                             QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen

class GridButton(QPushButton):
    """自定义按钮，支持存储和显示多个标签编号"""
    def __init__(self, row, col):
        super().__init__()
        self.row = row
        self.col = col
        self.label_ids = set()
        
        # [核心修改 1] 删除 setFixedSize(40, 40)
        # [核心修改 2] 设置尺寸策略为自动扩展，使其撑满网格并对齐上方的标签栏
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 可以设置一个最小尺寸，防止窗口被缩得太小导致界面崩溃
        self.setMinimumSize(20, 20)
        
        # 让按钮忽略鼠标事件，全交给主窗口处理
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.update_appearance()

    def add_label(self, label_id):
        if label_id:
            self.label_ids.add(label_id)
            self.update_appearance()

    def remove_label(self, label_id):
        if label_id in self.label_ids:
            self.label_ids.remove(label_id)
            self.update_appearance()

    def update_appearance(self):
        if self.label_ids:
            labels_str = ",".join(str(lbl) for lbl in sorted(list(self.label_ids)))
            self.setText(labels_str)
            self.setStyleSheet("""
                background-color: rgba(100, 200, 255, 150); 
                border: 1px solid blue; 
                text-align: top left;
                padding: 2px;
                font-weight: bold;
                font-size: 11px;
                color: black;
            """)
        else:
            self.setText("")
            self.setStyleSheet("background-color: rgba(255, 255, 255, 50); border: 1px solid gray;")

class SelectionGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.current_selected_label = None
        self.origin = QPoint()
        self.selection_rect = QRect()
        self.is_selecting = False
        self.selection_mode = None 
        self.buttons = []
        
        self.drag_position = None
        self.init_ui()

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.9)

        # 稍微放大一点初始窗口尺寸，以便于展示自适应效果
        self.resize(500, 900)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        self.create_title_bar()

        # 创建标签选择栏
        label_layout = QHBoxLayout()
        for i in range(1, 9):
            btn = QPushButton(f"标签 {i}")
            # 给顶部的标签栏也加上一点高度，显得更协调
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, idx=i: self.select_label(idx))
            label_layout.addWidget(btn)
        self.main_layout.addLayout(label_layout)

        # 创建 20行 x 10列 网格
        grid_layout = QGridLayout()
        # 强制水平和垂直间距一致为2像素
        grid_layout.setHorizontalSpacing(2)
        grid_layout.setVerticalSpacing(2)
        
        for r in range(20):
            row_btns = []
            for c in range(10):
                btn = GridButton(r, c)
                grid_layout.addWidget(btn, r, c)
                row_btns.append(btn)
            self.buttons.append(row_btns)
        
        self.main_layout.addLayout(grid_layout)
        # [优化] 设置布局中各部分的拉伸比例，让网格区域占据最大的空间
        self.main_layout.setStretchFactor(grid_layout, 1)
        self.setLayout(self.main_layout)

    def create_title_bar(self):
        title_bar_layout = QHBoxLayout()
        self.title_label = QLabel(" 拖动这里移动窗口 (支持自适应大小与防重叠)")
        self.title_label.setStyleSheet("color: white; font-weight: bold; background-color: rgba(50, 50, 50, 200); padding: 5px;")
        self.title_label.setMinimumHeight(30)
        
        close_btn = QPushButton("关闭")
        close_btn.setFixedSize(50, 30)
        close_btn.setStyleSheet("background-color: #d9534f; color: white; border: none; font-weight: bold;")
        close_btn.clicked.connect(self.close)
        
        title_bar_layout.addWidget(self.title_label)
        title_bar_layout.addWidget(close_btn)
        self.main_layout.addLayout(title_bar_layout)

    def select_label(self, idx):
        self.current_selected_label = idx
        print(f"当前选中标签: {idx}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_label.geometry().contains(event.pos()):
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return

        if self.current_selected_label:
            if event.button() == Qt.LeftButton:
                self.origin = event.pos()
                self.selection_rect = QRect(self.origin, event.pos())
                self.is_selecting = True
                self.selection_mode = 'select'
            elif event.button() == Qt.RightButton:
                self.origin = event.pos()
                self.selection_rect = QRect(self.origin, event.pos())
                self.is_selecting = True
                self.selection_mode = 'deselect'

    def mouseMoveEvent(self, event):
        if self.drag_position and not self.is_selecting:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
        elif self.is_selecting:
            self.selection_rect = QRect(self.origin, event.pos()).normalized()
            self.update() 

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            if self.is_selecting:
                self.apply_selection()
                self.is_selecting = False
                self.selection_mode = None
                self.selection_rect = QRect()
                self.update()
        
        if event.button() == Qt.LeftButton:
            self.drag_position = None

    def check_overlap(self):
        """检查当前的框选矩形是否碰到了已经有标签的按钮"""
        if self.selection_mode != 'select':
            return False 
            
        for row in self.buttons:
            for btn in row:
                if self.selection_rect.intersects(btn.geometry()) or btn.geometry().contains(self.origin):
                    if btn.label_ids: 
                        return True
        return False

    def apply_selection(self):
        if self.selection_mode == 'select' and self.check_overlap():
            print("选区与已有组重叠，操作取消！")
            return

        for row in self.buttons:
            for btn in row:
                if self.selection_rect.intersects(btn.geometry()) or btn.geometry().contains(self.origin):
                    if self.selection_mode == 'select':
                        btn.add_label(self.current_selected_label)
                    elif self.selection_mode == 'deselect':
                        btn.remove_label(self.current_selected_label)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20, 80))
        
        if self.is_selecting:
            is_overlapping = self.check_overlap()
            
            if self.selection_mode == 'select':
                if is_overlapping:
                    painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
                    painter.setBrush(QColor(255, 100, 0, 80))
                else:
                    painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine))
                    painter.setBrush(QColor(0, 255, 255, 50))
            elif self.selection_mode == 'deselect':
                painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
                painter.setBrush(QColor(255, 0, 0, 50))
                
            painter.drawRect(self.selection_rect)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SelectionGrid()
    window.show()
    sys.exit(app.exec_())