import sys
import json
import numpy as np
from plyfile import PlyData
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    
    for point in building.points:
        if user_shape.is_point_inside(point):
            collision_points.append(point.tolist())  # Convert to list for JSON serialization
    
    if collision_points:
        return collision_points, None, None
    else:
        distance, index = building.kdtree.query(user_shape.parameters['position'])
        nearest_point = building.points[index].tolist()  # Convert to list for JSON serialization
        return [], nearest_point, distance

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
            
            self.visualize_results(building, user_shape, collision_points, nearest_point, distance)
        
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")
    
    def visualize_results(self, building, user_shape, collision_points, nearest_point, distance):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot building points
        ax.scatter(building.points[:, 0], building.points[:, 1], building.points[:, 2], c='lightblue', s=1, alpha=0.5)
        
        # Plot user shape
        if user_shape.shape_type == 'box':
            corners = np.array(list(itertools.product(*zip(user_shape.parameters['position'], 
                                                           user_shape.parameters['position'] + user_shape.parameters['dimensions']))))
            ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='red', s=50)
        elif user_shape.shape_type == 'sphere':
            center = user_shape.parameters['position']
            ax.scatter(*center, c='red', s=50)
            # Add wireframe for sphere
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + user_shape.parameters['radius'] * np.cos(u) * np.sin(v)
            y = center[1] + user_shape.parameters['radius'] * np.sin(u) * np.sin(v)
            z = center[2] + user_shape.parameters['radius'] * np.cos(v)
            ax.plot_wireframe(x, y, z, color="r", alpha=0.5)
        # TODO: Add visualization for other shapes
        
        # Plot collision points or nearest point
        if len(collision_points) > 0:
            collision_points = np.array(collision_points)
            ax.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2], c='yellow', s=30)
            plt.title(f"Collision Detected: {len(collision_points)} points")
        elif nearest_point is not None:
            ax.scatter(*nearest_point, c='green', s=50)
            plt.title(f"No Collision. Nearest Point Distance: {distance:.2f}")
            # Draw line to nearest point
            ax.plot([user_shape.parameters['position'][0], nearest_point[0]],
                    [user_shape.parameters['position'][1], nearest_point[1]],
                    [user_shape.parameters['position'][2], nearest_point[2]], 'g--')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save the plot as a PNG file
        plt.savefig('collision_visualization.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CollisionDetectorGUI()
    ex.show()
    sys.exit(app.exec_())