import sys
import json
import numpy as np
from plyfile import PlyData, PlyElement
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QComboBox, QFileDialog)
from scipy.spatial import cKDTree

class BuildingInterior:
    def __init__(self, ply_path):
        self.plydata = PlyData.read(ply_path)
        self.points = np.column_stack((self.plydata['vertex']['x'], self.plydata['vertex']['y'], self.plydata['vertex']['z']))
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

    def get_shape_points(self, resolution=10):
        if self.shape_type == 'box':
            min_corner = self.parameters['position']
            max_corner = min_corner + self.parameters['dimensions']
            x = np.linspace(min_corner[0], max_corner[0], resolution)
            y = np.linspace(min_corner[1], max_corner[1], resolution)
            z = np.linspace(min_corner[2], max_corner[2], resolution)
            return np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        elif self.shape_type == 'sphere':
            center = self.parameters['position']
            radius = self.parameters['radius']
            phi = np.linspace(0, np.pi, resolution)
            theta = np.linspace(0, 2*np.pi, resolution)
            x = center[0] + radius * np.outer(np.sin(phi), np.cos(theta)).flatten()
            y = center[1] + radius * np.outer(np.sin(phi), np.sin(theta)).flatten()
            z = center[2] + radius * np.outer(np.cos(phi), np.ones_like(theta)).flatten()
            return np.column_stack((x, y, z))
        elif self.shape_type == 'cylinder':
            center = self.parameters['position']
            radius = self.parameters['radius']
            height = self.parameters['height']
            axis = self.parameters['axis']
            theta = np.linspace(0, 2*np.pi, resolution)
            z = np.linspace(0, height, resolution)
            theta, z = np.meshgrid(theta, z)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
            rotation_matrix = np.array([axis, np.cross([0, 0, 1], axis), [0, 0, 1]]).T
            points = np.dot(points, rotation_matrix) + center
            return points
        elif self.shape_type == 'pyramid':
            base_center = self.parameters['position']
            base_size = self.parameters['base_size']
            height = self.parameters['height']
            x = np.linspace(-base_size/2, base_size/2, resolution)
            y = np.linspace(-base_size/2, base_size/2, resolution)
            z = np.linspace(0, height, resolution)
            base_points = np.array(np.meshgrid(x, y, [0])).T.reshape(-1, 3)
            apex_point = np.array([[0, 0, height]])
            points = np.vstack((base_points, apex_point))
            return points + base_center
        return np.array([])

def check_collision(building, user_shape, tolerance=0.1):
    collision_points = []
    
    for point in building.points:
        if user_shape.is_point_inside(point):
            collision_points.append(point.tolist())
    
    if collision_points:
        return collision_points, None, None
    else:
        distance, index = building.kdtree.query(user_shape.parameters['position'])
        nearest_point = building.points[index].tolist()
        return [], nearest_point, float(distance)  # Convert distance to float

class CollisionDetectorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # PLY file path input
        ply_layout = QHBoxLayout()
        ply_layout.addWidget(QLabel('PLY File:'))
        self.ply_path_input = QLineEdit()
        ply_layout.addWidget(self.ply_path_input)
        self.browse_button = QPushButton('Browse')
        self.browse_button.clicked.connect(self.browse_file)
        ply_layout.addWidget(self.browse_button)
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
        
        # Connect shape combo box to update dimension inputs
        self.shape_combo.currentIndexChanged.connect(self.update_dimension_inputs)
        
    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select PLY File", "", "PLY Files (*.ply)")
        if filename:
            self.ply_path_input.setText(filename)
    
    def update_dimension_inputs(self):
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
            self.dim_input = self.axis_input
        elif shape == 'pyramid':
            self.dim_layout.addWidget(QLabel('Base Size:'))
            self.base_input = QLineEdit('1')
            self.dim_layout.addWidget(self.base_input)
            self.dim_layout.addWidget(QLabel('Height:'))
            self.height_input = QLineEdit('1')
            self.dim_input = self.height_input
        
        self.dim_layout.addWidget(self.dim_input)
    
    def check_collision(self):
        try:
            ply_path = self.ply_path_input.text()
            building = BuildingInterior(ply_path)
            
            shape = self.shape_combo.currentText()
            position = [float(x) for x in self.pos_input.text().split(',')]  # Convert to list
            
            if shape == 'box':
                dimensions = [float(x) for x in self.dim_input.text().split(',')]  # Convert to list
                parameters = {'position': position, 'dimensions': dimensions}
            elif shape == 'sphere':
                radius = float(self.dim_input.text())
                parameters = {'position': position, 'radius': radius}
            elif shape == 'cylinder':
                radius = float(self.radius_input.text())
                height = float(self.height_input.text())
                axis = [float(x) for x in self.axis_input.text().split(',')]  # Convert to list
                parameters = {'position': position, 'radius': radius, 'height': height, 'axis': axis}
            elif shape == 'pyramid':
                base_size = float(self.base_input.text())
                height = float(self.height_input.text())
                parameters = {'position': position, 'base_size': base_size, 'height': height}
            
            user_shape = UserShape(shape, parameters)
            collision_points, nearest_point, distance = check_collision(building, user_shape)
            
            # Prepare results dictionary
            results = {
                "collision": len(collision_points) > 0,
                "shape": shape,
                "parameters": parameters
            }
            
            if len(collision_points) > 0:
                results["collision_points"] = collision_points
                self.result_label.setText(f"Collision detected! {len(collision_points)} points in collision.")
            else:
                results["nearest_point"] = nearest_point
                results["distance"] = distance
                self.result_label.setText(f"No collision detected. Nearest point distance: {distance:.2f}")
            
            # Save results as JSON
            with open('collision_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Add user shape to PLY and save as new file
            self.add_shape_to_ply(building, user_shape)
            
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")
    
    def add_shape_to_ply(self, building, user_shape):
        # Get points for the user shape
        shape_points = user_shape.get_shape_points()
        
        # Combine original points with shape points
        all_points = np.vstack((building.points, shape_points))
        
        # Create a new vertex element
        vertex = np.array([(p[0], p[1], p[2]) for p in all_points], 
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        
        # Create a new PlyData object
        new_plydata = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
        
        # Save the new PLY file
        output_path = self.ply_path_input.text().replace('.ply', '_with_shape.ply')
        new_plydata.write(output_path)
        self.result_label.setText(self.result_label.text() + f"\nNew PLY saved as: {output_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CollisionDetectorGUI()
    ex.show()
    sys.exit(app.exec_())