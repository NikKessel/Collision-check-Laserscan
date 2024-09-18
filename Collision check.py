import sys
import numpy as np
from plyfile import PlyData
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox
from scipy.spatial import cKDTree

class BuildingInterior:
    def __init__(self, ply_path):
        plydata = PlyData.read(ply_path)
        self.points = np.column_stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']))
        self.kdtree = cKDTree(self.points)

class UserShape:
    def __init__(self, shape_type, parameters):
        self.shape_type = shape_type
        self.parameters = parameters
    
    def is_point_inside(self, point):
        if self.shape_type == 'box':
            min_corner = self.parameters['position']
            max_corner = min_corner + self.parameters['dimensions']
            return np.all(point >= min_corner) and np.all(point <= max_corner)
        elif self.shape_type == 'sphere':
            center = self.parameters['position']
            radius = self.parameters['radius']
            return np.linalg.norm(point - center) <= radius
        elif self.shape_type == 'cylinder':
            center = self.parameters['position']
            radius = self.parameters['radius']
            height = self.parameters['height']
            axis = self.parameters['axis']
            point_centered = point - center
            projection = np.dot(point_centered, axis)
            return (np.linalg.norm(point_centered - projection * axis) <= radius and 
                    0 <= projection <= height)
        elif self.shape_type == 'pyramid':
            base_center = self.parameters['position']
            base_size = self.parameters['base_size']
            height = self.parameters['height']
            point_centered = point - base_center
            x, y, z = point_centered
            return (abs(x) <= base_size * (1 - z / height) / 2 and
                    abs(y) <= base_size * (1 - z / height) / 2 and
                    0 <= z <= height)
        return False

def check_collision(building, user_shape, tolerance=0.1):
    collision_points = []
    
    # Check each point in the building
    for point in building.points:
        if user_shape.is_point_inside(point):
            collision_points.append(point)
    
    # Check for near-collisions
    if not collision_points:
        near_points = building.kdtree.query_ball_point(user_shape.parameters['position'], tolerance)
        collision_points = [building.points[i] for i in near_points]
    
    return collision_points

class CollisionDetectorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # PLY file path input
        ply_layout = QHBoxLayout()
        ply_layout.addWidget(QLabel('PLY File Path:'))
        self.ply_path_input = QLineEdit(r"C:\Users\KesselN\Downloads\prediction_scene1000_00.ply")
        ply_layout.addWidget(self.ply_path_input)
        layout.addLayout(ply_layout)
        
        # Shape selection
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel('Shape:'))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(['box', 'sphere', 'cylinder', 'pyramid'])
        shape_layout.addWidget(self.shape_combo)
        layout.addLayout(shape_layout)
        
        # Dimension inputs
        self.dim_layout = QVBoxLayout()
        layout.addLayout(self.dim_layout)
        self.update_dimension_inputs()
        
        # Position input
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel('Position (x,y,z):'))
        self.pos_input = QLineEdit('0,0,0')
        pos_layout.addWidget(self.pos_input)
        layout.addLayout(pos_layout)
        
        # Check collision button
        self.check_button = QPushButton('Check Collision')
        self.check_button.clicked.connect(self.check_collision)
        layout.addWidget(self.check_button)
        
        # Result display
        self.result_label = QLabel('')
        layout.addWidget(self.result_label)
        
        self.setLayout(layout)
        self.setWindowTitle('Collision Detector')
        self.show()
        
        # Connect shape combo box to update dimension inputs
        self.shape_combo.currentIndexChanged.connect(self.update_dimension_inputs)
        
    def update_dimension_inputs(self):
        # Clear existing inputs
        for i in reversed(range(self.dim_layout.count())): 
            self.dim_layout.itemAt(i).widget().setParent(None)
        
        shape = self.shape_combo.currentText()
        if shape == 'box':
            self.dim_layout.addWidget(QLabel('Dimensions (w,h,d):'))
            self.dim_input = QLineEdit('1,1,1')
        elif shape == 'sphere':
            self.dim_layout.addWidget(QLabel('Radius:'))
            self.dim_input = QLineEdit('1')
        elif shape == 'cylinder':
            self.dim_layout.addWidget(QLabel('Radius:'))
            self.radius_input = QLineEdit('1')
            self.dim_layout.addWidget(self.radius_input)
            self.dim_layout.addWidget(QLabel('Height:'))
            self.height_input = QLineEdit('1')
            self.dim_layout.addWidget(self.height_input)
            self.dim_layout.addWidget(QLabel('Axis (x,y,z):'))
            self.axis_input = QLineEdit('0,0,1')
            self.dim_input = self.axis_input  # For consistency in check_collision method
        elif shape == 'pyramid':
            self.dim_layout.addWidget(QLabel('Base Size:'))
            self.base_input = QLineEdit('1')
            self.dim_layout.addWidget(self.base_input)
            self.dim_layout.addWidget(QLabel('Height:'))
            self.height_input = QLineEdit('1')
            self.dim_input = self.height_input  # For consistency in check_collision method
        
        self.dim_layout.addWidget(self.dim_input)
    
    def check_collision(self):
        try:
            ply_path = self.ply_path_input.text()
            building = BuildingInterior(ply_path)
            
            shape = self.shape_combo.currentText()
            position = np.array([float(x) for x in self.pos_input.text().split(',')])
            
            if shape == 'box':
                dimensions = np.array([float(x) for x in self.dim_input.text().split(',')])
                parameters = {'position': position, 'dimensions': dimensions}
            elif shape == 'sphere':
                radius = float(self.dim_input.text())
                parameters = {'position': position, 'radius': radius}
            elif shape == 'cylinder':
                radius = float(self.radius_input.text())
                height = float(self.height_input.text())
                axis = np.array([float(x) for x in self.axis_input.text().split(',')])
                parameters = {'position': position, 'radius': radius, 'height': height, 'axis': axis}
            elif shape == 'pyramid':
                base_size = float(self.base_input.text())
                height = float(self.height_input.text())
                parameters = {'position': position, 'base_size': base_size, 'height': height}
            
            user_shape = UserShape(shape, parameters)
            collisions = check_collision(building, user_shape)
            
            if collisions:
                self.result_label.setText(f"Collision detected! {len(collisions)} points in collision.")
            else:
                self.result_label.setText("No collision detected.")
        
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CollisionDetectorGUI()
    sys.exit(app.exec_())